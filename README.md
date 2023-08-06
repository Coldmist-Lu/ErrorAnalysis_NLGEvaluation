# ErrorAnalysis_NLGEvaluation
<b>Toward Human-Like Evaluation for Natural Language Generation with Error Analysis (ACL2023)</b><br>

ErrorAnalysis_NLGEvaluation is a NLG evaluation metric computed by analyse the explicit/ implicit errors in the hypothesis. This is interpretable and human-like since we display how the metric refine the hypothesis into a better sentence.

We release the Metric, BARTScore4NLG, including the scripts of error analysis and scoring + prompting process for the replication of this study.

Usage in Python Command:

```python
from score import BARTScore4NLG_Scorer

scorer = BARTScore4NLG_Scorer(task='MT_WMT20', setting='zh-en', signature='bs:4|model:para')

src = ['Mike goes to the bookstore.', 'The cat is on the mat.']
tgt = ['Jerry goes to bookstore happily.', 'The mat sat on the mat.']
scorer.score(src, tgt)
```

See test.ipynb for more information. The scripts are 

If you find this work helpful, please consider citing as follows:

```ruby
@inproceedings{lu-etal-2023-toward,
    title = "Toward Human-Like Evaluation for Natural Language Generation with Error Analysis",
    author = "Lu, Qingyu  and
      Ding, Liang  and
      Xie, Liping  and
      Zhang, Kanjian  and
      Wong, Derek F.  and
      Tao, Dacheng",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.324",
    pages = "5892--5907",
}
```





