using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RL_Settings : MonoBehaviour
{
    public GameObject Agent;
    public GameObject Target;

    private Vector3 agentInitPos;
    private Vector3 targetInitPos;

    private Transform agentTrans;
    private Transform targetTrans;

    // Start is called before the first frame update
    void Start()
    {
        agentTrans = Agent.transform;
        targetTrans = Target.transform;
        
        agentInitPos = agentTrans.position;
        targetInitPos = targetTrans.position;


        
    }

    public void AreaSetting()
    {
        agentTrans.position = agentInitPos;
        targetTrans.position = targetInitPos;
        
    }
}
