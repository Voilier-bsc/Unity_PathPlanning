using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Linq;

public class RL_agent : Agent
{
    public Transform targetTrans;
    public LayerMask unwalkable;
    private Vector3 outPos;

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

    private Vector3 rnd_init_start;
    private Vector3 rnd_init_target;
    AGrid grid;

    float Dist;
    float preDist;

    float Dist_reward_p = 1;

    // int targetIndex = 0;
    


    public override void Initialize()
    {
        base.Initialize();

        grid = GetComponent<AGrid>();

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
    }

    public override void OnActionReceived(ActionBuffers actionsBuffers)
    {
        AddReward(-0.01f);

        var actions = actionsBuffers.ContinuousActions;

        Rays.Clear();
        Ray_Dist.Clear();

        float action_X = transform.position.x + Mathf.Clamp(actions[0], -1f, 1f) * 10;
        float action_Y = transform.position.z + Mathf.Clamp(actions[1], -1f, 1f) * 10;
        outPos = new Vector3(action_X, 0, action_Y);
        transform.position = Vector3.MoveTowards(transform.position,outPos,speed * Time.deltaTime);

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


        Dist = Vector3.Distance(targetTrans.position, transform.position);

        // target에 도착한 경우
        if(Dist <= 3.5f){
            SetReward(30f);
            EndEpisode();
        }
        // 지역을 벗어난 경우
        else if(transform.position.x > maxArea || transform.position.x < minArea || transform.position.z > maxArea || transform.position.z < minArea){
            SetReward(-30f);
            EndEpisode();
        }
        // 충돌이 일어난 경우
        else if(Ray_Dist.Min() <= robot_radius){
            SetReward(-30f);
            EndEpisode();
        }
        else{
            float dist_reward = preDist - Dist;
            AddReward(dist_reward/Dist_reward_p);
            preDist = Dist;
            // Debug.Log(dist_reward/Dist_reward_p);
        }

        // target에 가까워지는 경우
        

        // local path를 잘 따라가는 경우
        
        

        //reward 설정
        //AddReward() : 한스텝의 결과로 도출되는 보상에 입력값을 더함
        //SetReward() : 한스텝의 결과로 도출되는 보상을 입력값으로 결정

        //obstacle과 닿는 경우 SetReward(-1)
        //거리에 따른 보상 - 거의 도착한 경우 SetReward(1)
        // 
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
			// targetIndex = 1;
		}
	}

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ContinuousActionsOut = actionsOut.ContinuousActions;

        ContinuousActionsOut[0] = Input.GetAxis("Vertical");
        ContinuousActionsOut[1] = Input.GetAxis("Horizontal");
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



}

