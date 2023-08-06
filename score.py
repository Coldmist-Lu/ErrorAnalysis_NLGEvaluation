from BARTScore.bart_score_plusplus import BARTScorer_plusplus

# Task: Setting: (variant, prompt_loc, weight_lambda)
BS4NLG_SETTINGS = {
    'MT_WMT20': {
        'cs-en': ('f', 'enc', 0.80),
        'de-en': ('f', 'enc', 0.40),
        'ja-en': ('f', 'enc', 0.50),
        'ru-en': ('f', 'enc', 1.70),
        'zh-en': ('f', 'enc', 1.10),
        'iu-en': ('f', 'enc', 0.95),
        'km-en': ('f', 'enc', 1.30),
        'pl-en': ('f', 'enc', 0.85),
        'ps-en': ('f', 'enc', 1.10),
        'ta-en': ('f', 'enc', 0.60),
    },
    'SUM_REALSumm': {
        'COV': ('r', 'dec', 0.95),
    },
    'SUM_SummEval': {
        'COH': ('p', 'dec', 1.00),
        'FAC': ('p', 'dec', 0.75),
        'FLU': ('p', 'dec', 1.40),
        'INFO': ('p', 'dec', 0.95),
    },
    'SUM_NeR18': {
        'COH': ('p', 'dec', 1.10),
        'FLU': ('p', 'dec', 0.75),
        'INFO': ('p', 'dec', 0.70),
        'REL': ('p', 'dec', 0.70),
    },
    'SUM_Rank19': ('p', 'no', 0.85),
    'SUM_QAGS_CNN': ('p', 'no', 1.00),
    'SUM_QAGS_XSUM': ('p', 'no', 0.90),
    'D2T_BAGEL': ('f', 'dec', 2.00),
    'D2T_SFRES': ('f', 'dec', 1.40),
    'D2T_SFHOT': ('f', 'dec', 4.90),
}

PROMPT_CONTEXTS_PATHS = {
    'MT_WMT20': './prompt_file/mt_prompt.txt',
    'SUM_REALSumm': './prompt_file/sumh2r_prompt.txt',
    'SUM_SummEval': './prompt_file/sums2h_prompt.txt',
    'SUM_NeR18': './prompt_file/sums2h_prompt.txt',
    'SUM_Rank19': './prompt_file/sums2h_prompt.txt',
    'SUM_QAGS_CNN': './prompt_file/sums2h_prompt.txt',
    'SUM_QAGS_XSUM': './prompt_file/sums2h_prompt.txt',
    'D2T_BAGEL': './prompt_file/d2t_prompt.txt',
    'D2T_SFRES': './prompt_file/d2t_prompt.txt',
    'D2T_SFHOT': './prompt_file/d2t_prompt.txt',
}

class BARTScore4NLG_Scorer():
    def __init__(self, signature, task=None, setting=None, model=None, ckpt_path=None, prompt_path=None, weight_lambda=None):

        """
        task: the task to evaluate. If you are using BARTScore for new tasks, other parameters(like weight_lambda) should be specified.
        setting: the specific task setting if you provide the task. See the BS4NLG_SETTINGS above to know the settings.
        signature: we recommend using signature (dev:0|bs:4|model:para) to set the hyperparameters accordingly. We use the same format as in SacreBLEU.
        model: if you use models other than BART, specify the model path. You can also use other models in huggingface.
        ckpt_path: the checkpoint you use. Since vanilla BARTScore provide a variant called BART-PARA. You can download this .pth model and specify using this parameter.
        prompt_path: if you use your own prompt list, specify the path of file(.txt) in this parameter.
        weight_lambda: if you use your own lambda as the ratio of weights (explicit /implicit errors), specify in this parameter.
        """
    
        if weight_lambda == None:
            # use global setting.
            assert task in BS4NLG_SETTINGS.keys(), f'ensure the task name in: {list(BS4NLG_SETTINGS.keys())}'
            if setting:
                assert setting in BS4NLG_SETTINGS[task].keys(), f'ensure the task setting in: {list(BS4NLG_SETTINGS[task].keys())}'
            else:
                assert type(BS4NLG_SETTINGS[task]) == tuple, f'the setting of the task is no assigned!'
            variant, prompt_loc, weight_lambda = BS4NLG_SETTINGS[task][setting] if setting else BS4NLG_SETTINGS[task]

        self.scorer = BARTScorer_plusplus()
        self.scorer.set_signature(f'var:{variant}|prompt:{prompt_loc}|lambda:{weight_lambda}')
        self.scorer.set_signature(signature)
        self.scorer.load_model(model=model, path=ckpt_path)
        if prompt_path:
            self.scorer.load_prompt_contexts(path=prompt_path)
        else:
            assert task != None, 'the task is not specified! we do not know which prompt_lists to use.'
            self.scorer.load_prompt_contexts(path=PROMPT_CONTEXTS_PATHS[task])

    def score(self, src, tgt):
        """
        src: references for most scenarios, in some summarization tasks mean source passage.
        tgt: hypothesis or candidates to be evaluated.
        """
        return self.scorer.BARTScore_plus(src, tgt)



    
