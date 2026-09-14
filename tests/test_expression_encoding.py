from types import SimpleNamespace as NS
import expression_encoding as ee
def test_alpha_bend_and_direct_fields():
 n={"links":{"hammer_pull_destination_id":2},"techniques":{"palm_mute":True,"vibrato":{"name":"Slight"},"bend_type":{"name":"BendRelease"},"bend_points":[{"offset":0,"value":0},{"offset":60,"value":4}]}}
 o=ee.encode_alphatab(n,{"fret":5,"sus":2},7,{"rhythm":{"pick_stroke":{"name":"Down"}}})
 assert o["ho"] and o["pm"] and o["vb"] and o["pkd"]==0 and o["bn"]==2 and o["bnv"][-1]["t"]==2
