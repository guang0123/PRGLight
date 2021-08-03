# PRGLight

## Summary

PRGLight is a traffic light control algorithm. In PRGLight, we combine traffic prediction and traffic light control to jointly control the traffic light phase and traffic light duration.

## Algorithm

- For the traffic light phase control, we design a novel pressure index, Pressure with Remaining Capacity of the Outgoing Lane (PRCOL), defined as $$ PRCOL = N_{in}\times \left( 1-N_{out}/N_{max}  \right), $$ where $N_{in} $ and $N_{out} $ are the number of vehicles on the incoming and outgoing lane respectively, $N_{max} $ is the capacity of the outgoing lane. We further design a Reinforcement Learning agent with PRCOL as reward. We adopt the structure of [CoLight](https://sites.psu.edu/huawei/2019/09/15/colight-cikm-2019/).
- For the traffic light duration control, we adopt a GNN module STGCN ([paper](https://www.ijcai.org/proceedings/2018/0505), [code](https://github.com/VeritasYin/STGCN_IJCAI-18)). The duration is then decided by the light phase from the RL agent and the predicted traffic volume from the GNN. We design a flexible function to calculate the light duration from the real-time and future traffic condition. 
