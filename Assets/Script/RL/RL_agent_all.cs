using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Linq;
using System;
public class RL_agent_all : Agent
{
    public Transform targetTrans;
    public Transform waypointTrans;
    public LayerMask unwalkable;
    private Vector3 outPos;

    LineRenderer lr;

    public float maxArea = 25;
    public float minArea = -25;
    public float maxRay = 20;
    public float robot_radius = 1;
    public int way_interval = 5;
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
    private Vector3 area_pos;
    AGrid grid;

    float Dist;
    float preDist;

    float Dist_reward_p = 5;
    float Dist_Gloabl_reward_p = 1;

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

        area_pos = gameObject.transform.parent.parent.gameObject.transform.position;
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
        sensor.AddObservation(currentWaypoint.x);
        sensor.AddObservation(currentWaypoint.z);
    }

    public override void OnActionReceived(ActionBuffers actionsBuffers)
    {
        // Debug.Log(GetCumulativeReward());
        AddReward(-0.1f);

        if (globalpath.Length != 0){
            if(targetIndex >= globalpath.Length){
                currentWaypoint = globalpath[globalpath.Length-1];
            }
            else{
                currentWaypoint = globalpath[targetIndex];
            }
            waypointTrans.position = currentWaypoint + new Vector3(0,1,0);
        }

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
            // case k_UpRight:
            //     transform.position = transform.position + UpRight;
            //     break;
            // case k_UpLeft:
            //     transform.position = transform.position + UpLeft;
            //     break;
            // case k_DownRight:
            //     transform.position = transform.position + DownRight;
            //     break;
            // case k_DownLeft:
            //     transform.position = transform.position + DownLeft;
            //     break;
            
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


        if (Vector3.Distance(currentWaypoint, transform.position) <= 2){
            AddReward(Dist_Gloabl_reward_p*targetIndex);
            targetIndex ++;
        }


        Dist = Vector3.Distance(targetTrans.position, transform.position);
       
        // Debug.Log(StepCount);

        Debug.Log(Ray_Dist.Min());
        // target에 도착한 경우
        if(Dist <= 3.5f){
            SetReward(10f);
            Debug.Log(GetCumulativeReward());
            EndEpisode();
        }

        // 충돌이 일어난 경우
        
        else if(Ray_Dist.Min() <= robot_radius){
            SetReward(-10f);
            Debug.Log(GetCumulativeReward());
            EndEpisode();
        }

        else{
            // target에 가까워지는 경우
            float dist_reward = preDist - Dist;
            AddReward(dist_reward/Dist_reward_p);
            preDist = Dist;
        }
        // Debug.Log(GetCumulativeReward());
        

        //reward 설정
        //AddReward() : 한스텝의 결과로 도출되는 보상에 입력값을 더함
        //SetReward() : 한스텝의 결과로 도출되는 보상을 입력값으로 결정

    }

    public override void OnEpisodeBegin()
    {
        
        
		while(true){
			rnd_init_start = area_pos + new Vector3((int)UnityEngine.Random.Range(minArea + robot_radius, maxArea - robot_radius),transform.position.y,(int)UnityEngine.Random.Range(minArea + robot_radius, maxArea - robot_radius));
			rnd_init_target = area_pos + new Vector3((int)UnityEngine.Random.Range(minArea + robot_radius, maxArea - robot_radius),targetTrans.position.y,(int)UnityEngine.Random.Range(minArea + robot_radius, maxArea - robot_radius));
			if((!Physics.CheckSphere(rnd_init_start, 4, unwalkable))&& (!Physics.CheckSphere(rnd_init_target, 4, unwalkable)) && (Vector3.Distance(rnd_init_start, rnd_init_target) > 30)){
				transform.position = rnd_init_start;
				targetTrans.position = rnd_init_target;
				break;
			}
		}
        preDist = Vector3.Distance(targetTrans.position, transform.position);


        FindPath(transform.localPosition, targetTrans.localPosition);
        targetIndex = 0;
        // GlobalPathRequestManager.RequestPath(transform.position,targetTrans.position, OnPathFound);
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
	
	void FindPath(Vector3 startPos, Vector3 targetPos) {

		Vector3[] waypoints = new Vector3[0];
		bool pathSuccess = false;
		
		Node startNode = grid.NodeFromWorldPoint(startPos);
		Node targetNode = grid.NodeFromWorldPoint(targetPos);
		
		
		if (startNode.walkable && targetNode.walkable) {
			Heap<Node> openSet = new Heap<Node>(grid.MaxSize);
			HashSet<Node> closedSet = new HashSet<Node>();
			openSet.Add(startNode);
			
			while (openSet.Count > 0) {
				Node currentNode = openSet.RemoveFirst();
				closedSet.Add(currentNode);
				
				if (currentNode == targetNode) {
					pathSuccess = true;
					break;
				}
				
				foreach (Node neighbour in grid.GetNeighbours(currentNode)) {
					if (!neighbour.walkable || closedSet.Contains(neighbour)) {
						continue;
					}
					
					int newMovementCostToNeighbour = currentNode.gCost + GetDistance(currentNode, neighbour);
					if (newMovementCostToNeighbour < neighbour.gCost || !openSet.Contains(neighbour)) {
						neighbour.gCost = newMovementCostToNeighbour;
						neighbour.hCost = GetDistance(neighbour, targetNode);
						neighbour.parent = currentNode;
						
						if (!openSet.Contains(neighbour))
							openSet.Add(neighbour);
					}
				}
			}
		}

		if (pathSuccess) {
			waypoints = RetracePath(startNode,targetNode);
		}

        globalpath = waypoints;
		
	}

	// 탐색 종료 후 최종 노드의 ParentNode를 추적하며 리스트에 담는다.
	// 최종 경로에 있는 노드들의 worldPosition을 순차적으로 담아서 return
	Vector3[] RetracePath(Node startNode, Node endNode) {
		List<Node> path = new List<Node>();
		Node currentNode = endNode;
		
		while (currentNode != startNode) {
			path.Add(currentNode);
			currentNode = currentNode.parent;
		}
		Vector3[] waypoints = SimplifyPath(path);
		Array.Reverse(waypoints);
		return waypoints;
		
	}
	
	// Path list에 있는 노드들의 worldPosition을 Vector3[] 배열에 담아서 return
	Vector3[] SimplifyPath(List<Node> path) { 
		List<Vector3> waypoints = new List<Vector3>();
		Vector2 directionOld = Vector2.zero;

		waypoints.Add(path[0].worldPosition);
		for (int i = 1; i < path.Count; i ++) {
			Vector2 directionNew = new Vector2(path[i-1].gridX - path[i].gridX,path[i-1].gridY - path[i].gridY);
			// if (directionNew != directionOld) { //direction이 바뀐 경우에만 waypoint에 추가
			// waypoints.Add(path[i].worldPosition);
			// }
			
			if (i%way_interval == 0) {
				waypoints.Add(path[i].worldPosition);
			}

			directionOld = directionNew;
		}
		
		return waypoints.ToArray();
	}
	
	int GetDistance(Node nodeA, Node nodeB) {
		int dstX = Mathf.Abs(nodeA.gridX - nodeB.gridX);
		int dstY = Mathf.Abs(nodeA.gridY - nodeB.gridY);
		
		if (dstX > dstY)
			return 14*dstY + 10* (dstX-dstY);
		return 14*dstX + 10 * (dstY-dstX);
	}



}

