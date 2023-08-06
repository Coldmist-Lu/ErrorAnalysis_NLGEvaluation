import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
from tqdm import tqdm

class BARTScorer_plusplus:
    def __init__(self, 
                device='cuda:0', 
                max_length=1024, # max length for segments
                model_name='bartpara', # 'bartlarge', 'bartcnn', 'bartpara'.
                variant='f', # 'f': F score; 'p': Precision; 'r': Recall
                batch_size=4, # batch size
                sample_topk=10, # k for selecting best token
                weight_lambda = 1.0,
                prompt_loc='no', # no: w/o prompt; enc: prompt suffxed to src; dec: prompt prefixed to tgt.
                nt_method='overlap', # method for checking non-translation error
                ):

        # Set up model
        self.device = device
        self.max_length = max_length
        self.model_name = model_name
        self.variant = variant
        self.batch_size = batch_size
        self.sample_topk = sample_topk
        self.weight_lambda = weight_lambda
        self.prompt_loc = prompt_loc
        self.nt_method = nt_method

    def load_model(self, model=None, path=None):
        """Init model before scoring or correction. """
        assert self.model_name in ['bartlarge', 'bartcnn', 'bartpara']
        if model:
            checkpoint = model
        elif self.model_name == 'bartlarge':
            checkpoint = 'facebook/bart-large'
        elif self.model_name in ['bartcnn', 'bartpara']:
            checkpoint = 'facebook/bart-large-cnn'
        # Construct BART model to GPU device

        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(self.device)

        # Set up loss functions
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

        # Load model from paraphrase finetuning
        if self.model_name == 'bartpara':
            if path is None:
                path = 'models/bart.pth'
            self.model.load_state_dict(torch.load(path, map_location=self.device))

    def print_params(self):
        """Print parameters of BARTScore++ Metric. """
        print('---------------Metric Parameters-------------------')
        print('Model name: ', self.model_name)
        print('Device: ', self.device)
        print('Batch size: ', self.batch_size)
        print('Token Max Length: ', self.max_length)
        print('K in top-k sampling: ', self.sample_topk)
        print('Prompt Location: ', self.prompt_loc)
        print('Method for Checking Non-translation: ', self.nt_method)
        print('BARTScore Variant: ', self.variant)
        return

    def set_signature(self, signature: str):
        """Changing metric parameters using signature. """
        # eg: signature = 'name:bartpara|dev:0|var:f|bs:10|topk:10|prompt:enc|nt:model'
        params = signature.split('|')
        for p in params:
            if p:
                parameter = p.split(':')[-1]
            if p.startswith('name'): # model_name: bart-para(default)
                self.model_name = parameter
            if p.startswith('dev'): # using which decive: cuda:0(default)
                self.device = 'cuda:' + parameter
            if p.startswith('var'): # score method
                self.variant = parameter.lower()
            if p.startswith('maxlen'): # max token length
                self.max_length = int(parameter)
            if p.startswith('bs'): # batch_size for correction: 4(default)
                self.batch_size = int(parameter)
            if p.startswith('topk'): # topk: 10(default)
                self.sample_topk = int(parameter)
            if p.startswith('lambda'): # weighted lambda
                self.weight_lambda = float(parameter)
            if p.startswith('prompt'): # prompt method: no/enc/dec
                self.prompt_loc = parameter
            if p.startswith('nt'): # non translation method: overlap or model
                self.ntc_method = parameter

        return
        
    def load_prompt_contexts(self, path):
        """Load prompt contexts. """
        def read_file_to_list(file_name):
            """Read prompt contexts file. """
            lines = []
            with open(file_name, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    lines.append(line.strip())
            return lines

        self.prompt_contexts_list = read_file_to_list(path)

    def tokenize(self, sents):
        """tokenize sents"""
        batch = self.tokenizer(sents, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt')
        return {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'len': batch['attention_mask'].sum(dim=1).to(self.device),
        }

    def compute_score(self, src, tgt):
        """Compute average_loss -> score of vanilla BARTScore (src->tgt). """
        label_ids = tgt['input_ids']
        src_input = {'input_ids': src['input_ids'], 'attention_mask': src['attention_mask']}
        output = self.model(labels=label_ids, **src_input)
        logits = output.logits.view(-1, self.model.config.vocab_size)
        loss = self.loss_fct(self.lsm(logits), label_ids.view(-1))
        loss = loss.view(label_ids.shape[0], -1)
        loss_avg = loss.sum(dim=1) / tgt['len']
        return np.array([-x.item() for x in loss_avg])

    def compute_variant_score(self, src, tgt, variant=''):
        """Compute BARTScore variant: precision, recall or f score"""
        variant = self.variant if variant == '' else variant # if variant is not assigned, then use global parameter.
        assert variant in ['f', 'p', 'r'], "the bartscore variant parameter should be 'f', 'p', or 'r']. "
        if variant == 'f':
            src2tgt = self.compute_score(src, tgt) # compute precision
            tgt2src = self.compute_score(tgt, src) # compute recall
            return (src2tgt + tgt2src) / 2 # average
        elif variant == 'p':
            return self.compute_score(src, tgt)
        elif variant == 'r':
            return self.compute_score(tgt, src)

    def assign_prompt(self, srcs, tgts, p, prompt_loc, variant=''):
        """ assign prompt based on methods. """
        def add_prompt(l, method='no'):
            """ add prompt to sentences. Method - suffix (on encoder) or prefix (on decoder) """
            assert method in ['suffix', 'prefix', 'no']
            if method == 'no':
                return l
            if method == 'suffix':
                return [x + ' ' + p + ',' for x in l]
            if method == 'prefix':
                return [p + ', ' + x for x in l]
            
        variant = self.variant if variant == '' else variant # if variant is not explicitly assigned, then use global parameter.
        assert prompt_loc in ['no', 'enc', 'dec'], "Prompt location should be 'no', 'enc' or 'dec'. "
        assert variant in ['p', 'r'], "BARTScore variant should be 'p' or 'r'. "
        PROMPT_ASSIGN_DICT = {
            ('enc', 'p'): ('suffix', 'no'),
            ('enc', 'r'): ('no', 'suffix'),
            ('dec', 'p'): ('no', 'prefix'),
            ('dec', 'r'): ('prefix', 'no'),
        }
        method_src, method_tgt = PROMPT_ASSIGN_DICT[(prompt_loc, variant)]
        # assign prompt
        return add_prompt(srcs, method_src), add_prompt(tgts, method_tgt)
    
    def compute_score_withprompt(self, srcs, tgts, prompt_context, prompt_loc=''):
        """Calculate score with prompt. """
        assert isinstance(srcs, list) and isinstance(tgts, list), 'Please ensure input data are in "list" format.'
        prompt_loc = self.prompt_loc if prompt_loc == '' else prompt_loc
        if self.variant == 'f':
            # precision
            tmp_srcs, tmp_tgts = self.assign_prompt(srcs, tgts, prompt_context, prompt_loc, 'p')
            score_p = self.compute_variant_score(self.tokenize(tmp_srcs), self.tokenize(tmp_tgts), 'p')
            # recall
            tmp_srcs, tmp_tgts = self.assign_prompt(srcs, tgts, prompt_context, prompt_loc, 'r')
            score_r = self.compute_variant_score(self.tokenize(tmp_srcs), self.tokenize(tmp_tgts), 'r')
            # average
            return (score_p + score_r) / 2
        else:
            tmp_srcs, tmp_tgts = self.assign_prompt(srcs, tgts, prompt_context, prompt_loc)
            return self.compute_variant_score(srcs, tgts)

    def check_multi_references(self, refs, tgts):
        """Check if all references are in list or str format. """

        # assert all references are in the same format - list or string.
        counter_ref = sum([isinstance(ref, list) for ref in refs])
        assert counter_ref in [len(refs), 0], 'All references should be in "list" or "string" format. '

        if counter_ref == 0: # single reference mode (all refs are in string format)
            return refs
        else: # multi refrences mode (all refs are in list format)
            print('Multi-references detected.')
            single_refs = []
            for i in tqdm(range(len(refs)), desc='Choose best reference', ncols=100):
                # choose the reference best related to 
                scores = []
                for j in range(0, len(refs[i]), self.batch_size):
                    refs_list = refs[i][j: j + self.batch_size]
                    scores.extend(self.compute_variant_score(refs_list, [tgts[i]] * len(refs_list)).tolist())
                best_index = np.argmax(scores)
                single_refs.append(refs[i][best_index])
            return single_refs

    def scoring_process(self, srcs, tgts, label='tgt'):
        """Perform the whole scoring process with all prompts. """
        assert len(srcs) == len(tgts)

        scores = {}

        # score with no prompt
        sig = label + ' no prompt'
        scores[sig] = []
        for i in tqdm(range(0, len(srcs), self.batch_size), desc='scoring ' + sig, ncols=100):
            src_list = srcs[i: i + self.batch_size]
            tgt_list = tgts[i: i + self.batch_size]
            scores[sig].extend(self.compute_variant_score(self.tokenize(src_list), self.tokenize(tgt_list)).tolist())

        # score with each prompt
        if self.prompt_loc != 'no':
            for context in self.prompt_contexts_list:
                sig = label + ' ' + self.prompt_loc + ' ' + context 
                scores[sig] = []
                for i in tqdm(range(0, len(srcs), self.batch_size), desc='scoring ' + sig, ncols=100):
                    src_list = srcs[i: i + self.batch_size]
                    tgt_list = tgts[i: i + self.batch_size]
                    scores[sig].extend(self.compute_score_withprompt(src_list, tgt_list, context).tolist())

        return scores
        
    def check_non_translation(self, src, tgt):
        """Perform Non-translation Check. Return True: non-trans; False: normal
           Here src, tgt should contain only 1 sample.
        """
        assert self.nt_method in ['overlap', 'model'], 'Non-translation method should be "overlap" or "model". '

        if self.nt_method == 'overlap' and tgt['len'] >= 10: # suited for most MT & D2T tasks.
            set1 = set(src['input_ids'].cpu().detach().tolist())
            set2 = set(tgt['input_ids'].cpu().detach().tolist())
            # if overlap ratio <= 0.1 -> non_translation
            return len(set1 & set2) / len(set1 | set2) <= 0.2

        else: # suited for SUM, and MT w. paraphrased references, and short sentences.
            tgt = self.convert_2D(tgt)
            src = self.convert_2D(src)
            label_ids = tgt['input_ids']
            src_input = {'input_ids': src['input_ids'], 'attention_mask': src['attention_mask']}
            output = self.model(**src_input, labels=label_ids)
            logits = output.logits.view(-1, self.model.config.vocab_size)
            loss = -self.loss_fct(self.lsm(logits), label_ids.view(-1))
            src2tgt = loss.sum() / tgt['len'][0]
            src2src = self.compute_variant_score(src, src, variant='p')[0]
            nt_score = (src2tgt + src2src) / 2
            # if more than 70% of token below ntr_score -> non_translation
            return len((loss < nt_score).nonzero().view(-1)) >= tgt['len'][0] * 0.7

    def create_edit_candidates(self, _tgt, tokens, index):
        tgt = self.convert_1D(_tgt)
        ids, mask, length = tgt['input_ids'], tgt['attention_mask'], tgt['len']

        edit_strategy = ['keep', 'delete'] + ['add'] * len(tokens) + ['replace'] * len(tokens)
        edit_token = torch.cat((tgt['input_ids'][index].view(-1), torch.zeros(1).to(self.device), tokens, tokens))
        # keep & delete
        cand_ids = [ids, torch.cat((ids[:index], ids[index+1:]))]
        cand_masks = [mask, mask[1:]]
        cand_len = [length, length-1]
        # add
        cand_ids.extend([torch.cat((ids[:index], t.view(1), ids[index:])) for t in tokens])
        cand_masks.extend([torch.cat((torch.ones(1, dtype=torch.int64).to(self.device), tgt['input_ids']))] * len(tokens))
        cand_len.extend([length + 1] * len(tokens))
        # replace
        cand_ids.extend([torch.cat((ids[:index], t.view(1), ids[index+1:])) for t in tokens])
        cand_masks.extend([mask] * len(tokens))
        cand_len.extend([length] * len(tokens))

        return [{
            'strategy': strategy,
            'token': token, 
            'cand': {
                'input_ids': ids,
                'attention_mask': mask,
                'len': length,
            }
        } for strategy, token, ids, mask, length in zip(edit_strategy, edit_token, cand_ids, cand_masks, cand_len)]

    def error_detect_correct(self, _src, _tgt):
        """correct error. """
        from torch.nn.utils.rnn import pad_sequence
        
        # get logits and loss
        tgt = self.convert_2D(_tgt)
        src = self.convert_2D(_src)
        label_ids = tgt['input_ids']
        src_input = {'input_ids': src['input_ids'], 'attention_mask': src['attention_mask']}
        output = self.model(**src_input, labels=label_ids)
        logits = output.logits.view(-1, self.model.config.vocab_size)
        loss = -self.loss_fct(self.lsm(logits), label_ids.view(-1))

        # detected token index
        index = torch.argmin(loss)

        # get top-k candidate tokens
        tokens = torch.topk(logits[index], k=self.sample_topk, sorted=False).indices
        tokens = tokens[tokens > 2]

        tgt = self.convert_1D(tgt)
        src = self.convert_1D(src)

        # edit sentences
        edit_cands = self.create_edit_candidates(tgt, tokens, index)

        # compute batch score
        cand_score_list = []
        for i in range(0, len(edit_cands), self.batch_size):
            cands = edit_cands[i: i+self.batch_size]
            batch_len = len(cands)
            cur_src = {
                'input_ids': src['input_ids'].repeat(batch_len, 1),
                'attention_mask': src['attention_mask'].repeat(batch_len, 1),
                'len': src['len'].repeat(batch_len)
            }
            cur_tgt = {
                'input_ids': pad_sequence([cand['cand']['input_ids'] for cand in cands],
                                          batch_first=True, padding_value=1),
                'attention_mask': pad_sequence([cand['cand']['attention_mask'] for cand in cands],
                                               batch_first=True, padding_value=0),
                'len': torch.cat([cand['cand']['len'].view(1) for cand in cands]),
            }
            cand_score_list.extend(self.compute_variant_score(cur_src, cur_tgt).tolist())

        # select best
        best_id = np.argmax(cand_score_list)
        if edit_cands[best_id]['token'] == edit_cands[0]['token']: # if keep
            best_id = 0

        # return token index, edit candidate information, and the score after editing.
        return index, edit_cands[best_id], cand_score_list[best_id]
    

    def decode(self, input_ids, token_level=False):
        """Decode input ids into sentences. """

        if token_level == True:
            # decode word by word
            if input_ids.dim() == 0:
                return self.tokenizer.decode(input_ids.view(1), skip_special_tokens=True).strip()
            return [self.tokenizer.decode(token.view(1), skip_special_tokens=True).strip() for token in input_ids]
        else:
            # decode the whole sentence
            sent = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            if sent == '':
                return ''
            else:
                return sent[0].upper() + sent[1:] if len(sent) > 1 else sent[0].upper()


    def refine_sentence(self, srcs, tgts):
        """Performing Self-correction operation using bartscore_forward, return correction information. """

        srcs = self.check_multi_references(srcs, tgts)

        refine_list = []
        for i in tqdm(range(len(srcs)), desc='Refine sentences', ncols=100):
            try:
                with torch.no_grad():
                    # tokenize
                    src = self.convert_1D(self.tokenize(srcs[i]))
                    tgt = self.convert_1D(self.tokenize(tgts[i]))

                    # refine information
                    refine_dict = {
                        'src': src,
                        'tgt': tgt, 
                        'iteration': -1,
                        'error analysis': [] 
                    }

                    # non-translation check
                    if not self.check_non_translation(src, tgt):
                        # refine sentence
                        max_iter = np.ceil(tgt['len'].cpu().detach().tolist() * 0.5).astype('int')
                        for iters in range(max_iter):
                            index, edit_dict, edit_score = self.error_detect_correct(src, tgt)
                            if edit_dict['strategy'] == 'keep':
                                # break iteration.
                                refine_dict['iteration'] = iters
                                break
                            # record
                            refine_dict['error analysis'].append({
                                'sent': edit_dict['cand']['input_ids'],
                                'index': index,
                                'strategy': edit_dict['strategy'],
                                'token_before': tgt['input_ids'][index],
                                'token_refine': edit_dict['token'],
                                'score': edit_score,
                            })
                            # update tgt
                            tgt = edit_dict['cand']
                        else:
                            refine_dict['iteration'] = max_iter

                    refine_list.append(refine_dict)

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {srcs[i]}')
                print(f'target: {tgts[i]}')
                exit(0) 

        return refine_list, srcs


    def post_processing(self, srcs, tgts, refine_list):
        """Calculating scores for final BARTScore++."""

        # decode sentence
        rfi_sents = []
        for refine_dict in refine_list:
            refine_dict['src'] = self.decode(refine_dict['src']['input_ids'])
            refine_dict['tgt'] = self.decode(refine_dict['tgt']['input_ids'])
            for info in refine_dict['error analysis']:
                info['sent'] = self.decode(info['sent'])
                info['token_before'] = self.decode(info['token_before'], token_level=True)
                info['token_refine'] = self.decode(info['token_refine'], token_level=True)
            rfi_sents.append(refine_dict['error analysis'][-1]['sent'])

        # calculate scores
        s2s_scores = self.scoring_process(srcs, srcs, 'src')
        s2r_scores = self.scoring_process(srcs, rfi_sents, 'refine')
        s2t_scores = self.scoring_process(srcs, tgts, 'tgt')

        # record scores
        for id, refine_dict in enumerate(refine_list):
            refine_dict.update({
                's2s_score': {sig: score[id] for sig, score in s2s_scores.items()},
                's2r_score': {sig: score[id] for sig, score in s2r_scores.items()},
                's2t_score': {sig: score[id] for sig, score in s2t_scores.items()},
            })

        return refine_list              


    def weighted_sum_score(self, res):

        # if lambda >= 10, we only use explicit distance
        w_exp = self.weight_lambda / (1+self.weight_lambda) if self.weight_lambda < 10 else 1
        w_imp = 1 / (1+self.weight_lambda) if self.weight_lambda < 10 else 0

        final_scores = []
        for refine_dict in res:
            if self.prompt_loc == 'no':
                s2s_score = refine_dict['s2s_score']['src no prompt']
                s2r_score = refine_dict['s2r_score']['refine no prompt']
                s2t_score = refine_dict['s2t_score']['tgt no prompt']
            else:
                s2s_score = np.average([score for sig, score in refine_dict['s2s_score'].items() if sig != 'src no prompt'])
                s2r_score = np.average([score for sig, score in refine_dict['s2r_score'].items() if sig != 'refine no prompt'])
                s2t_score = np.average([score for sig, score in refine_dict['s2t_score'].items() if sig != 'tgt no prompt'])

            exp_dist, imp_dist = s2s_score - s2r_score, s2r_score - s2t_score
            final_scores.append(-exp_dist * w_exp - imp_dist * w_imp)

        return final_scores
    
        
    def BARTScore_plus(self, srcs, tgts):
        self.print_params()
        res, srcs = self.refine_sentence(srcs, tgts)
        res = self.post_processing(srcs, tgts, res)
        scores = self.weighted_sum_score(res)
        return scores, res


    @staticmethod
    def convert_2D(input_dict):
        """convert 1D tokenized tensors(input_ids, attention_mask) into 2D.""" 
        if input_dict['input_ids'].dim() == 2:
            return input_dict   
        return {
            'input_ids': input_dict['input_ids'].view(1, -1),
            'attention_mask': input_dict['attention_mask'].view(1, -1),
            'len': input_dict['len'].view(-1)
        }
    

    @staticmethod
    def convert_1D(input_dict):
        """convert 2D tokenized tensors(input_ids, attention_mask) into 1D."""
        if input_dict['input_ids'].dim() == 1:
            return input_dict   
        return {
            'input_ids': input_dict['input_ids'].view(-1),
            'attention_mask': input_dict['attention_mask'].view(-1),
            'len': input_dict['len'][0]
        }



# %%

# %%
