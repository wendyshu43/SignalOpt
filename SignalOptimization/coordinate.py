import os, sys
from numpy import mean
import numpy as np
import pandas as pd
sys.path.append('C:/Users/WendySHU/PycharmProjects/multiRamp')
import dataProcess.timeSeriesCorrelation as tsc


def coordinate(timeStamp):
    # 初始化
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    import traci
    sumoBinary = "D:/sumo-1.2.0/bin/sumo"
    sumoConfig = "D:/Acadamy/sumo/multi_ramp/mr.sumocfg"
    sumoCmd = [sumoBinary, "-c", sumoConfig]

    # 控制变量
    detector_interval = 120
    ramp_control_cycle = 60
    ramp_signal_cycle = 60
    # 协调控制变量
    inflowCount1 = 0
    inflowCount2 = 0
    inflowCount3 = 0
    inflowCount4 = 0

    outflowCount1 = 0
    outflowCount2 = 0
    outflowCount3 = 0
    outflowCount4 = 0

    rampinflow1 = 0
    rampinflow2 = 0
    rampinflow3 = 0
    rampinflow4 = 0

    bottleneck_threshold = 90
    qcap = 80  # 88
    sindex = 2.727

    # 结果列表 TTT、TWT、Density、Velocity、Flow、Queue
    travelTime = []
    waitingTime = []
    waitingTime1 = []
    waitingTime2 = []
    waitingTime3 = []
    waitingTime4 = []

    queueList = []
    queueList1 = []
    queueList2 = []
    queueList3 = []
    queueList4 = []

    densityList1 = []
    densityList2 = []
    densityList3 = []
    densityList4 = []

    velocityList1 = []
    velocityList2 = []
    velocityList3 = []
    velocityList4 = []

    flowCount = 0

    lastPhaseDuration = []

    # 每一步的存储值
    stepTravelTime= []

    stepWaitingTime = []
    stepTWT1 = []
    stepTWT2 = []
    stepTWT3 = []
    stepTWT4 = []

    stepDen1 = []
    stepDen2 = []
    stepDen3 = []
    stepDen4 = []

    stepQue = []
    stepQue1 =[]
    stepQue2 = []
    stepQue3 = []
    stepQue4 = []

    stepVel1 = []
    stepVel2 = []
    stepVel3 = []
    stepVel4 = []

    # 打开traci接口
    traci.start(sumoCmd)

    # 开始进行仿真
    step = 0
    while step < timeStamp:
        # 旅行时间的范围：2300m，最大速度22.22m/s，所以旅行时间最少要104s，最大值预设400
        ttt = traci.edge.getTraveltime("m_in") + traci.edge.getTraveltime("m_bn1") + traci.edge.getTraveltime("m_f1") + \
              traci.edge.getTraveltime("m_bn2") + traci.edge.getTraveltime("m_f2") + traci.edge.getTraveltime("m_bn3") + \
              traci.edge.getTraveltime("m_f3") + traci.edge.getTraveltime("m_bn4") + traci.edge.getTraveltime("m_out")

        if 104 < ttt < 400:
            stepTravelTime.append(ttt)

        # 等待时间
        twt1 = traci.edge.getWaitingTime("r_r1")
        if 0 < twt1 < 60:
            waitingTime1.append(twt1)
        twt2 = traci.edge.getWaitingTime("r_r2")
        if 0 < twt2 < 60:
            waitingTime2.append(twt2)
        twt3 = traci.edge.getWaitingTime("r_r3")
        if 0 < twt3 < 60:
            waitingTime3.append(twt3)
        twt4 = traci.edge.getWaitingTime("r_r4")
        if 0 < twt4 < 60:
            waitingTime4.append(twt4)

        twt = twt1 + twt2 + twt3 + twt4
        if 0 < twt < 100:
            stepWaitingTime.append(twt)

        # 匝道排队长度
        que1 = (traci.lanearea.getJamLengthMeters("ramp1_0") + traci.lanearea.getJamLengthMeters("ramp1_1")) / 2
        if 0 <= que1 < 273:
            queueList1.append(que1)

        que2 = (traci.lanearea.getJamLengthMeters("ramp2_0") + traci.lanearea.getJamLengthMeters("ramp2_1")) / 2
        if 0 <= que1 < 273:
            queueList2.append(que2)

        que3 = (traci.lanearea.getJamLengthMeters("ramp3_0") + traci.lanearea.getJamLengthMeters("ramp3_1")) / 2
        if 0 <= que3 < 273:
            queueList3.append(que3)

        que4 = (traci.lanearea.getJamLengthMeters("ramp4_0") + traci.lanearea.getJamLengthMeters("ramp4_1")) / 2
        if 0 <= que4 < 273:
            queueList4.append(que4)

        que = (que1 + que2 + que3 + que4) / 4
        if 0 <= que < 273:
            stepQue.append(que)

        # 密度
        den1 = (traci.lanearea.getLastStepVehicleNumber("bn1_0") + traci.lanearea.getLastStepVehicleNumber("bn1_1") +
              traci.lanearea.getLastStepVehicleNumber("bn1_2") + traci.lanearea.getLastStepVehicleNumber("bn1_3") +
              traci.lanearea.getLastStepVehicleNumber("bn1_4") + traci.lanearea.getLastStepVehicleNumber("bn1_5") +
               traci.lanearea.getLastStepVehicleNumber("bn1_6")) / 0.1
        stepDen1.append(den1)

        den2 = (traci.lanearea.getLastStepVehicleNumber("bn2_0") + traci.lanearea.getLastStepVehicleNumber("bn2_1") +
              traci.lanearea.getLastStepVehicleNumber("bn2_2") + traci.lanearea.getLastStepVehicleNumber("bn2_3") +
              traci.lanearea.getLastStepVehicleNumber("bn2_4") + traci.lanearea.getLastStepVehicleNumber("bn2_5") +
               traci.lanearea.getLastStepVehicleNumber("bn2_6")) / 0.1
        stepDen2.append(den2)

        den3 = (traci.lanearea.getLastStepVehicleNumber("bn3_0") + traci.lanearea.getLastStepVehicleNumber("bn3_1") +
              traci.lanearea.getLastStepVehicleNumber("bn3_2") + traci.lanearea.getLastStepVehicleNumber("bn3_3") +
              traci.lanearea.getLastStepVehicleNumber("bn3_4") + traci.lanearea.getLastStepVehicleNumber("bn3_5") +
               traci.lanearea.getLastStepVehicleNumber("bn3_6")) / 0.1
        stepDen3.append(den3)

        den4 = (traci.lanearea.getLastStepVehicleNumber("bn4_0") + traci.lanearea.getLastStepVehicleNumber("bn4_1") +
              traci.lanearea.getLastStepVehicleNumber("bn4_2") + traci.lanearea.getLastStepVehicleNumber("bn4_3") +
              traci.lanearea.getLastStepVehicleNumber("bn4_4") + traci.lanearea.getLastStepVehicleNumber("bn4_5") +
               traci.lanearea.getLastStepVehicleNumber("bn4_6")) / 0.1
        stepDen4.append(den4)

        # 速度
        vel1 = (traci.inductionloop.getLastStepMeanSpeed("bn1_outflow_0") +
                traci.inductionloop.getLastStepMeanSpeed("bn1_outflow_1") +
                traci.inductionloop.getLastStepMeanSpeed("bn1_outflow_2")) / 3 * 3.6  # 速度也可以用edge, 3.6换算为kmh
        if 0 < vel1 < 80:
            stepVel1.append(vel1)

        vel2 = (traci.inductionloop.getLastStepMeanSpeed("bn2_outflow_0") +
                traci.inductionloop.getLastStepMeanSpeed("bn2_outflow_1") +
                traci.inductionloop.getLastStepMeanSpeed("bn2_outflow_2")) / 3 * 3.6  # 速度也可以用edge, 3.6换算为kmh
        if 0 < vel1 < 80:
            stepVel2.append(vel2)

        vel3 = (traci.inductionloop.getLastStepMeanSpeed("bn3_outflow_0") +
                traci.inductionloop.getLastStepMeanSpeed("bn3_outflow_1") +
                traci.inductionloop.getLastStepMeanSpeed("bn3_outflow_2")) / 3 * 3.6  # 速度也可以用edge, 3.6换算为kmh
        if 0 < vel3 < 80:
            stepVel3.append(vel3)

        vel4 = (traci.inductionloop.getLastStepMeanSpeed("bn4_outflow_0") +
                traci.inductionloop.getLastStepMeanSpeed("bn4_outflow_1") +
                traci.inductionloop.getLastStepMeanSpeed("bn4_outflow_2")) / 3 * 3.6  # 速度也可以用edge, 3.6换算为kmh
        if 0 < vel4 < 80:
            stepVel4.append(vel4)

        # 流量
        flow = traci.inductionloop.getLastStepVehicleNumber("main_outflow_0") + \
               traci.inductionloop.getLastStepVehicleNumber("main_outflow_1") + \
               traci.inductionloop.getLastStepVehicleNumber("main_outflow_2")
        flowCount += flow

        # 控制变量：进出流量
        inflow1 = traci.inductionloop.getLastStepVehicleNumber("bn1_inflow_0") + \
                  traci.inductionloop.getLastStepVehicleNumber("bn1_inflow_1") + \
                  traci.inductionloop.getLastStepVehicleNumber("bn1_inflow_2")
        inflowCount1 += inflow1

        outflow1 = traci.inductionloop.getLastStepVehicleNumber("bn1_outflow_0") + \
                    traci.inductionloop.getLastStepVehicleNumber("bn1_outflow_1") + \
                    traci.inductionloop.getLastStepVehicleNumber("bn1_outflow_2")
        outflowCount1 += outflow1

        # bn2

        inflow2 = traci.inductionloop.getLastStepVehicleNumber("bn2_inflow_0") + \
                  traci.inductionloop.getLastStepVehicleNumber("bn2_inflow_1") + \
                  traci.inductionloop.getLastStepVehicleNumber("bn2_inflow_2")
        inflowCount2 += inflow2

        outflow2 = traci.inductionloop.getLastStepVehicleNumber("bn2_outflow_0") + \
                   traci.inductionloop.getLastStepVehicleNumber("bn2_outflow_1") + \
                   traci.inductionloop.getLastStepVehicleNumber("bn2_outflow_2")
        outflowCount2 += outflow2
        # bn3

        inflow3 = traci.inductionloop.getLastStepVehicleNumber("bn3_inflow_0") + \
                  traci.inductionloop.getLastStepVehicleNumber("bn3_inflow_1") + \
                  traci.inductionloop.getLastStepVehicleNumber("bn3_inflow_2")
        inflowCount3 += inflow3

        outflow3 = traci.inductionloop.getLastStepVehicleNumber("bn3_outflow_0") + \
                   traci.inductionloop.getLastStepVehicleNumber("bn3_outflow_1") + \
                   traci.inductionloop.getLastStepVehicleNumber("bn3_outflow_2")
        outflowCount3 += outflow3
        # bn4

        inflow4 = traci.inductionloop.getLastStepVehicleNumber("bn4_inflow_0") + \
                  traci.inductionloop.getLastStepVehicleNumber("bn4_inflow_1") + \
                  traci.inductionloop.getLastStepVehicleNumber("bn4_inflow_2")
        inflowCount4 += inflow4

        outflow4 = traci.inductionloop.getLastStepVehicleNumber("bn4_outflow_0") + \
                   traci.inductionloop.getLastStepVehicleNumber("bn4_outflow_1") + \
                   traci.inductionloop.getLastStepVehicleNumber("bn4_outflow_2")
        outflowCount4 += outflow4

        # 瓶颈算法需要知道上一个控制周期匝道进入主线的流量
        rampinflow1 += traci.inductionloop.getLastStepVehicleNumber("bn1_inflow_3")
        rampinflow2 += traci.inductionloop.getLastStepVehicleNumber("bn2_inflow_3")
        rampinflow3 += traci.inductionloop.getLastStepVehicleNumber("bn3_inflow_3")
        rampinflow4 += traci.inductionloop.getLastStepVehicleNumber("bn4_inflow_3")

        # 协调控制算法
        if step % ramp_control_cycle == 0:

            # 权重矩阵需要提前给出，削减值也可以在之前提前给出，四个瓶颈都需要用到

            reduce1 = inflowCount1 + rampinflow1 - outflowCount1
            reduce2 = inflowCount2 + rampinflow2 - outflowCount2
            reduce3 = inflowCount3 + rampinflow3 - outflowCount3
            reduce4 = inflowCount4 + rampinflow4 - outflowCount4

            # weight = np.array([[1, 0.125, 0.052, 0.026],
            #                    [0, 0.875, 0.316, 0.184],
            #                    [0, 0, 0.632, 0.316],
            #                    [0, 0, 0, 0.474]])
            weight = tsc.dtw()

            # 先要判断是否是瓶颈
            # 第一个
            if mean(densityList1) > bottleneck_threshold:
                # 如果是瓶颈，需要计算协调控制率
                mrb1 = (rampinflow1 - max(reduce1 * weight[0, 0], reduce2 * weight[0, 1],
                                                  reduce3 * weight[0, 2], reduce4 * weight[0, 3])) * sindex
                mrl1 = (qcap - inflowCount1) * sindex
                rlocal1 = min(mrb1, mrl1)

            else:
                rlocal1 = (qcap - inflowCount1) * sindex

            # 第二个
            if mean(densityList2) > bottleneck_threshold:
                mrb2 = (rampinflow2 - max(reduce2 * weight[1, 1], reduce3 * weight[1, 2],
                                                  reduce4 * weight[1, 3])) * sindex
                mrl2 = (qcap - inflowCount2) * sindex
                rlocal2 = min(mrb2, mrl2)

            else:
                rlocal2 = (qcap - inflowCount2) * sindex

            # 第三个
            if mean(densityList3) > bottleneck_threshold:
                mrb3 = (rampinflow3 - max(reduce3*weight[2, 2], reduce4*weight[2, 3])) * sindex
                mrl3 = (qcap - inflowCount3) * sindex
                rlocal3 = min(mrb3, mrl3)

            else:
                rlocal3 = (qcap - inflowCount3) * sindex

            # 第四个
            if mean(densityList4) > bottleneck_threshold:
                mrb4 = (rampinflow4 - reduce4*weight[3, 3]) * sindex
                mrl4 = (qcap - inflowCount4) * sindex
                rlocal4 = min(mrb4, mrl4)

            else:
                rlocal4 = (qcap - inflowCount4) * sindex

            inflowCount1 = 0
            inflowCount2 = 0
            inflowCount3 = 0
            inflowCount4 = 0
            outflowCount1 = 0
            outflowCount2 = 0
            outflowCount3 = 0
            outflowCount4 = 0
            rampinflow1 = 0
            rampinflow2 = 0
            rampinflow3 = 0
            rampinflow4 = 0

            green_time_list = [rlocal1, rlocal2, rlocal3, rlocal4]
            actual_gt_list = [0, 0, 0, 0]

            flag = 0
            for gt in green_time_list:
                if gt > 0.9 * ramp_signal_cycle:
                    actual_gt_list[flag] = ramp_signal_cycle
                elif gt < 0.1 * ramp_signal_cycle:
                    actual_gt_list[flag] = 10
                else:
                    actual_gt_list[flag] = int(gt)
                flag += 1

            traci.trafficlight.setPhase("rl1", 0)
            traci.trafficlight.setPhaseDuration("rl1", actual_gt_list[0])
            traci.trafficlight.setPhase("rl2", 0)
            traci.trafficlight.setPhaseDuration("rl2", actual_gt_list[1])
            traci.trafficlight.setPhase("rl3", 0)
            traci.trafficlight.setPhaseDuration("rl3", actual_gt_list[2])
            traci.trafficlight.setPhase("rl4", 0)
            traci.trafficlight.setPhaseDuration("rl4", actual_gt_list[3])

            lastPhaseDuration.append(actual_gt_list)
            # print('\n', actual_gt_list, '\n')

        # 计算结果
        if step % detector_interval == 0:
            if stepTravelTime:
                travelTime.append(mean(stepTravelTime))
                stepTravelTime.clear()
            else:
                travelTime.append(0)

            if stepWaitingTime:
                waitingTime.append(mean(stepWaitingTime))
                stepWaitingTime.clear()
            else:
                waitingTime.append(0)

            if stepQue:
                queueList.append(mean(stepQue))
                stepQue.clear()
            else:
                queueList.append(0)

            if stepDen1:
                densityList1.append(mean(stepDen1))
                stepDen1.clear()
            else:
                densityList1.append(0)

            if stepDen2:
                densityList2.append(mean(stepDen2))
                stepDen2.clear()
            else:
                densityList2.append(0)

            if stepDen3:
                densityList3.append(mean(stepDen3))
                stepDen3.clear()
            else:
                densityList3.append(0)

            if stepDen4:
                densityList4.append(mean(stepDen4))
                stepDen4.clear()
            else:
                densityList4.append(0)

            if stepVel1:
                velocityList1.append(mean(stepVel1))
                stepVel1.clear()
            else:
                velocityList1.append(0)

            if stepVel2:
                velocityList2.append(mean(stepVel2))
                stepVel2.clear()
            else:
                velocityList2.append(0)

            if stepVel3:
                velocityList3.append(mean(stepVel3))
                stepVel3.clear()
            else:
                velocityList3.append(0)

            if stepVel4:
                velocityList4.append(mean(stepVel4))
                stepVel4.clear()
            else:
                velocityList4.append(0)

            if stepTWT1:
                waitingTime1.append(mean(stepTWT1))
                stepTWT1.clear()
            else:
                waitingTime1.append(0)

            if stepTWT2:
                waitingTime2.append(mean(stepTWT2))
                stepTWT2.clear()
            else:
                waitingTime2.append(0)

            if stepTWT3:
                waitingTime3.append(mean(stepTWT3))
                stepTWT3.clear()
            else:
                waitingTime3.append(0)

            if stepTWT4:
                waitingTime4.append(mean(stepTWT4))
                stepTWT4.clear()
            else:
                waitingTime4.append(0)

            if stepQue1:
                queueList1.append(mean(stepQue1))
                stepQue1.clear()
            else:
                queueList1.append(0)

            if stepQue2:
                queueList2.append(mean(stepQue2))
                stepQue2.clear()
            else:
                queueList2.append(0)

            if stepQue3:
                queueList3.append(mean(stepQue3))
                stepQue3.clear()
            else:
                queueList3.append(0)

            if stepQue4:
                queueList4.append(mean(stepQue4))
                stepQue4.clear()
            else:
                queueList4.append(0)

        traci.simulationStep()  # 运行一步仿真
        step += 1

    # 结束仿真
    traci.close()

    velDict = {'Bottleneck1': velocityList1, 'Bottleneck2': velocityList2, 'Bottleneck3': velocityList3, 'Bottleneck4': velocityList4}
    denDict = {'Bottleneck1': densityList1, 'Bottleneck2': densityList2, 'Bottleneck3': densityList3, 'Bottleneck4': densityList4}
    vel_df = pd.DataFrame.from_dict(velDict, orient='index').transpose()
    vel_df.to_csv('coorSpeed.csv', encoding='utf-8', index=True)
    den_df = pd.DataFrame.from_dict(denDict, orient='index').transpose()
    den_df.to_csv('coorDensity.csv', encoding='utf-8', index=True)

    resultDict = {'ttt': mean(travelTime), 'twt': mean(waitingTime), 'twt1': mean(waitingTime1),'twt2': mean(waitingTime2),
                  'twt3': mean(waitingTime3), 'twt4': mean(waitingTime4), 'que': mean(queueList), 'que1': mean(queueList1),
                  'que2': mean(queueList2), 'que3': mean(queueList3), 'que4': mean(queueList4), 'flow': flowCount,
                  'den1': mean(densityList1), 'den2': mean(densityList2), 'den3': mean(densityList3), 'den4': mean(densityList4),
                  'vel1': mean(velocityList1),'vel2': mean(velocityList2), 'vel3': mean(velocityList3), 'vel4': mean(velocityList4)}
    df_result = pd.DataFrame.from_dict(resultDict, orient='index').transpose()
    df_result.to_csv('result.csv', encoding='utf-8', index=True, mode='a', header=False)

    tttDict = {'coordinate': travelTime}
    df_ttt = pd.DataFrame.from_dict(tttDict, orient='index').transpose()
    df_ttt.to_csv('TTT.csv', encoding='utf-8', index=False, mode='a', header=True)

    return

#     return mean(travelTime), mean(waitingTime), mean(queueList), flowCount, \
#            mean(densityList1) + mean(densityList2) + mean(densityList3) + mean(densityList4)
#
#
# ttt, twt, que, flow, den = coordinate(10800)
# print('协调控制旅行时间：', ttt, '等待时间', twt, '排队长度：', que, '通过流量', flow, '瓶颈密度：', den)


coordinate(10800)

# resultcap = {}
#
# for cap in np.arange(70, 110, 1):
#     resultcap[cap] = coordinate(10800, cap, 90)
#
# for item in resultcap:
#     print("瓶颈容量：", item, "结果：", resultcap[item])

# resultbnt = {}
#
# for bnt in np.arange(110, 130, 1):
#     resultbnt[bnt] = coordinate(10800, bnt)
#
# for it in resultbnt:
#     print("瓶颈阈值：", it, "结果：", resultbnt[it])
