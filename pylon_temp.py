from pypower.api import case9,runpf
from runPfOptions import ppoption
import const


succes=runpf(case9(),ppopt=ppoption());
# print(succes[0]['branch'])
print(succes[0]['bus'])
for branch in succes[0]['branch']:
    print branch[const.constOut['branch']["Pin"]]
    print branch[const.constOut['branch']["Pout"]]
print("done")