using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Linq;

public class RL_multi_agent : Agent
{
    public Transform targetTrans;
    public LayerMask unwalkable;
    private Vector3 outPos;

    LineRenderer lr;

    public float maxArea = 50;
    public float minArea = -50;
    public float maxRay = 20;
    public float robot_radius = 1;

    public float speed = 10;
    
    public List<Ray> Rays = new List<Ray>();
    public List<Vector3> Rays_Vector = new List<Vector3>();
    public List<float> Ray_Dist = new List<float>();
    public RaycastHit hitData;

    Vector3[] globalpath;
    List<Vector3> globalpath_list = new List<Vector3>();
    List<float> cost_list = new List<float>();

    private Vector3 rnd_init_start;
    private Vector3 rnd_init_target;
    AGrid grid;

    float Dist;
    float preDist;

    float Dist_reward_p = 5;
    // float Dist_Gloabl_reward_p = 5;

    int targetIndex;
    Vector3 currentWaypoint;

    private Vector3 moveRight = new Vector3(1f,0,0f);
    private Vector3 moveLeft = new Vector3(-1f,0,0f);
    private Vector3 moveUp = new Vector3(0f,0,1f);
    private Vector3 moveDown = new Vector3(0f,0,-1f);
    private Vector3 UpRight = new Vector3(1f,0,1f);
    private Vector3 UpLeft = new Vector3(-1f,0,1f);
    private Vector3 DownRight = new Vector3(1f,0,-1f);
    private Vector3 DownLeft = new Vector3(-1f,0,-1f);
    
    const int k_NoAction = 0;  // do nothing!
    const int k_Up = 1;
    const int k_Down = 2;
    const int k_Left = 3;
    const int k_Right = 4;
    const int k_UpRight = 5;
    const int k_UpLeft = 6;
    const int k_DownRight = 7;
    const int k_DownLeft  = 8;


    public override void Initialize()
    {
        base.Initialize();

        grid = GetComponent<AGrid>();
        lr = GetComponent<LineRenderer>();

        Rays.Clear();
        Rays_Vector.Clear();
        Ray_Dist.Clear();


        for(int i = 0; i < 36; i++){
            Rays_Vector.Add(new Vector3(Mathf.Cos(Mathf.PI*i/10),0,Mathf.Sin(Mathf.PI*i/10)));
        }

        Academy.Instance.AgentPreStep += WaitTimeInference;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.position.x);
        sensor.AddObservation(transform.position.z);
        sensor.AddObservation(targetTrans.position.x);
        sensor.AddObservation(targetTrans.position.z);
        sensor.AddObservation(Vector3.Distance(targetTrans.position, transform.position));
        sensor.AddObservation(currentWaypoint.x);
        sensor.AddObservation(currentWaypoint.z);
        sensor.AddObservation(Vector3.Distance(transform.position, currentWaypoint));
    }

    public override void OnActionReceived(ActionBuffers actionsBuffers)
    {
        AddReward(-0.1f);

        if (globalpath != null){
            if(targetIndex >= globalpath.Length){
                currentWaypoint = globalpath[globalpath.Length-1];
            }
            else{
                currentWaypoint = globalpath[targetIndex];
            }
        }

        // AnimateLine();

        var actions = actionsBuffers.DiscreteActions[0];
        
        switch (actions)
        {
            case k_NoAction:
                // do nothing
                break;
            case k_Right:
                transform.position = transform.position + moveRight;
                break;
            case k_Left:
                transform.position = transform.position + moveLeft;
                break;
            case k_Up:
                transform.position = transform.position + moveUp;
                break;
            case k_Down:
                transform.position = transform.position + moveDown;
                break;
            case k_UpRight:
                transform.position = transform.position + UpRight;
                break;
            case k_UpLeft:
                transform.position = transform.position + UpLeft;
                break;
            case k_DownRight:
                transform.position = transform.position + DownRight;
                break;
            case k_DownLeft:
                transform.position = transform.position + DownLeft;
                break;
            
        }

        Rays.Clear();
        Ray_Dist.Clear();


        for(int i = 0; i < 36; i++){
            Rays.Add(new Ray(transform.position, Rays_Vector[i]));
        }

        for(int i = 0; i < 36; i++){     
            if(Physics.Raycast(Rays[i], out hitData, maxRay, unwalkable)){
                Ray_Dist.Add(hitData.distance);
            }
            else{
                Ray_Dist.Add(maxRay);
            }
        }


        // if (Vector3.Distance(currentWaypoint, transform.position) <= 1.5){
        //     AddReward(targetIndex);
        //     Debug.Log(GetCumulativeReward());
        //     targetIndex ++;
        // }


        Dist = Vector3.Distance(targetTrans.position, transform.position);
       
        // Debug.Log(StepCount);


        // target??? ????????? ??????
        if(Dist <= 3.5f){
            SetReward(30f);
            Debug.Log(GetCumulativeReward());
            EndEpisode();
        }
        // ????????? ????????? ??????
        else if(transform.position.x > maxArea || transform.position.x < minArea || transform.position.z > maxArea || transform.position.z < minArea){
            SetReward(-15f);
            EndEpisode();
        }
        // ????????? ????????? ??????
        else if(Ray_Dist.Min() <= robot_radius){
            SetReward(-15f);
            Debug.Log(GetCumulativeReward());
            EndEpisode();
        }

        else{
            // target??? ??????????????? ??????
            float dist_reward = preDist - Dist;
            AddReward(dist_reward/Dist_reward_p);
            preDist = Dist;
        }
        // Debug.Log(GetCumulativeReward());
        

        //reward ??????
        //AddReward() : ???????????? ????????? ???????????? ????????? ???????????? ??????
        //SetReward() : ???????????? ????????? ???????????? ????????? ??????????????? ??????

    }

    public override void OnEpisodeBegin()
    {
        
        
		while(true){
			rnd_init_start = new Vector3(Random.Range(-48,48),transform.position.y,Random.Range(-48,48));
			rnd_init_target = new Vector3(Random.Range(-48,48),targetTrans.position.y,Random.Range(-48,48));
			if((!Physics.CheckSphere(rnd_init_start, 4, unwalkable))&& (!Physics.CheckSphere(rnd_init_target, 4, unwalkable)) && (Vector3.Distance(rnd_init_start, rnd_init_target) > 10)){
				transform.position = rnd_init_start;
				targetTrans.position = rnd_init_target;
				break;
			}
		}
        
        
        preDist = Vector3.Distance(targetTrans.position, transform.position);

        GlobalPathRequestManager.RequestPath(transform.position,targetTrans.position, OnPathFound);
    }


    public void OnPathFound(Vector3[] newPath, bool pathSuccessful) {
        Debug.Log(pathSuccessful);
		if (pathSuccessful) {
			globalpath = newPath;
			targetIndex = 0;
		}
	}

    public override void Heuristic(in ActionBuffers actionsOut)
    {

        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = k_NoAction;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = k_Right;
        }
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = k_Up;
        }
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = k_Left;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = k_Down;
        }
    }

    public float DecisionWaitingTime = 5f;
    float m_currentTime = 0f;

    public void WaitTimeInference(int action){
        if(Academy.Instance.IsCommunicatorOn){
            RequestDecision();
        }
        else{
            if(m_currentTime >= DecisionWaitingTime){
                m_currentTime = 0f;
                RequestDecision();
            }
            else{
                m_currentTime += Time.fixedDeltaTime * 10;
            }
        }
    }

    private void AnimateLine(){
        if (globalpath != null) {
            if(targetIndex <= globalpath.Length){
                lr.positionCount = globalpath.Length-targetIndex;
                for (int i = 0; i < lr.positionCount; i++) {
                    
                    lr.SetPosition(i,globalpath[i + targetIndex]);
                }
            }
		}

        // if (globalpath != null) {
        //     lr.positionCount = globalpath.Length;
        //     for (int i = 0; i < lr.positionCount; i++) {
        //         lr.SetPosition(i,globalpath[i]);
        //     }
		// }
    }

}

