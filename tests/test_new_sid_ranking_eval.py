import unittest
from run_new_sid_ranking_eval import canonical_rows, validate_alignment


class NewSIDRankingTests(unittest.TestCase):
    def test_different_sid_strings_decode_to_same_real_ranking(self):
        source=[{'item_id':'1'}]
        old=canonical_rows({'rows':{'direct50':[dict(row_id=0,output='old1',predict=['old2','old1'])]}},
                           {'1':['old1'],'2':['old2']},source,[0])
        new=canonical_rows({'rows':{'direct50':[dict(row_id=0,output='new9',predict=['new3','new9'])]}},
                           {'1':['new9'],'2':['new3']},source,[0])
        self.assertEqual(old,new)
        with self.assertRaises(ValueError):
            canonical_rows({'rows':{'direct50':[dict(row_id=0,output='new3',predict=['new9'])]}},
                           {'1':['new9'],'2':['new3']},source,[0])

    def test_alignment_allows_only_sid_remapping(self):
        old=[dict(item_id='1',history_item_id='[2]',item_sid='a',history_item_sid='[b]')]
        new=[dict(item_id='1',history_item_id='[2]',item_sid='c',history_item_sid='[d]')]
        validate_alignment(old,new)
        new[0]['history_item_id']='[3]'
        with self.assertRaises(ValueError):validate_alignment(old,new)


if __name__ == '__main__':unittest.main()
