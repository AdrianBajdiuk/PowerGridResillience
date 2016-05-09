
##constanses for entering correct values into PF matrices
constIn={"bus":{"index":0,"type":1,"Pd":2,"Qd":3,"Vm":7,"Va":8,"baseKV":9},
         "gen":{"busIndex":0,"Pg":1,"Qg":2,"Qmax":3,"Qmin":4,"Vg":5,"Pmax":8,"Pmin":9},
         "branch":{"fromBus":0,"toBus":1,"r":2,"x":3,"b":4}
       }

## pf[0]['branch']
constOut={"branch":{"fromBus":0,"toBus":1,"Pin":13,"Qin":14,"Pout":15,"Qout":16},
          "bus":{"index":0,"Vm":7,"Va":8}
          }

simProcessorsCount = 1