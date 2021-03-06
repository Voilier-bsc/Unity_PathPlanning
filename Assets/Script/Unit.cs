using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Unit : MonoBehaviour {
	public Transform target;
	float speed = 10;
	Vector3[] path;
	Vector3[] localpath;
	int targetIndex;
    public LayerMask unwalkableMask;
    private Vector3 rnd_init_start;
    private Vector3 rnd_init_target;
    AGrid grid;
    public float robot_r = 1;
	Vector3 draw_vect = new Vector3(0f,0f,0f);



	/////////// local
	public int max_iter = 500; //500
    public int goal_sample_rate = 20; //5
    public int min_rand = -100;
    public int max_rand = 100;
    public float expand_dis = 2; // 3.0
    public float path_resolution = 0.1f; //0.5
    public float connect_circle_dist = 3; //50.0
    List<RRTNode> node_list = new List<RRTNode>();



    void Awake() {
        grid = GetComponent<AGrid>();
		while(true){
			rnd_init_start = new Vector3(Random.Range(-48,48),transform.position.y,Random.Range(-48,48));
			rnd_init_target = new Vector3(Random.Range(-48,48),target.position.y,Random.Range(-48,48));
			if((!Physics.CheckSphere(rnd_init_start, 4, unwalkableMask))&& (!Physics.CheckSphere(rnd_init_target, 4, unwalkableMask)) && (Vector3.Distance(rnd_init_start, rnd_init_target) > 10)){
				transform.position = rnd_init_start;
				target.position = rnd_init_target;
				break;
			}
		}
    }

	void Start() {
		GlobalPathRequestManager.RequestPath(transform.position,target.position, OnPathFound);
	}

	public void OnPathFound(Vector3[] newPath, bool pathSuccessful) {
		if (pathSuccessful) {
			path = newPath;
			targetIndex = 1;
			StopCoroutine("FollowPath");
			StartCoroutine("FollowPath");
		}
	}

	IEnumerator FollowPath() {
		Vector3 currentWaypoint = path[1];
        Vector3 prev_local_path = new Vector3(0,0,0);

		
		while (true) {
			// if (transform.position == currentWaypoint) {
			// 	targetIndex ++;

			// 	if (targetIndex >= path.Length) {
			// 		yield break;
			// 	}
			// 	currentWaypoint = path[targetIndex];
			// }

            if(Physics.CheckSphere(transform.position, robot_r + 0.1f, unwalkableMask)){
                Debug.Log("collision!");
            }
            
			localpath = localplanning(transform.position, currentWaypoint);

            if(targetIndex == path.Length){
                transform.position = Vector3.MoveTowards(transform.position, currentWaypoint, speed * Time.deltaTime);
                if(transform.position == currentWaypoint){
                    this.gameObject.layer = 0;
                    Debug.Log("Clear");
                    yield break;
                }
                yield return null;
                continue;
            }

            if(localpath != null){
                if(localpath.Length > 1){
                    transform.position = Vector3.MoveTowards(transform.position,localpath[1],speed * Time.deltaTime);
                    prev_local_path = localpath[1];
                }
                
                if(localpath.Length > 1 && localpath.Length < 5){
                    targetIndex ++;
                }

                if(targetIndex == path.Length){
                    yield return null;
                    continue;
                }
                currentWaypoint = path[targetIndex];
            }
			yield return null;
		}
	}

	Vector3 cube_range = new Vector3(1,1,1);
	public void OnDrawGizmos() {
		if (path != null) {
			for (int i = targetIndex; i < path.Length; i ++) {
				Gizmos.color = Color.black;
				Gizmos.DrawCube(path[i], Vector3.one);


				if (i == targetIndex) {
					Gizmos.DrawLine(transform.position, path[i]);
				}
				else {
					Gizmos.DrawLine(path[i-1],path[i]);
				}
			}
		}

        if(localpath!=null){
                foreach(Vector3 path in localpath){
                draw_vect = new Vector3(path.x, 1, path.z);
                Gizmos.color = Color.red;
                // Gizmos.DrawCube(draw_vect, cube_range);
            }
        }
	}






    public Vector3[] localplanning(Vector3 _startpos, Vector3 _goalpos){
		if(_startpos == _goalpos){
			return null;
		}

		// Debug.Log("try");


        node_list.Clear();
        RRTNode start_node = new RRTNode(_startpos.x, _startpos.z);
        RRTNode end_node = new RRTNode(_goalpos.x, _goalpos.z);


        node_list.Add(start_node);


        RRTNode rnd_node;
        int nearest_ind;
        RRTNode nearest_node;
        int last_index;
        Vector3[] local_path = new Vector3[0];
    
        for(int i = 0; i < max_iter; i++){
            rnd_node = get_random_node(end_node);
            nearest_ind = get_nearest_node_index(rnd_node);
            RRTNode New_Node = steer(node_list[nearest_ind], rnd_node, expand_dis);
            nearest_node = node_list[nearest_ind];

            New_Node.cost = nearest_node.cost + (Mathf.Pow(New_Node.x-nearest_node.x,2)+(Mathf.Pow(New_Node.y-nearest_node.y,2)));
            
            if (check_collision(New_Node)){
                List<int> near_inds = find_near_nodes(New_Node);
                RRTNode node_with_updated_parent = choose_parent(New_Node, near_inds);
                if(node_with_updated_parent != null){
                    rewire(node_with_updated_parent, near_inds);
                    node_list.Add(node_with_updated_parent);
                }
                else{
                    node_list.Add(New_Node);
                }
            }

            if(New_Node != null){
                last_index = search_best_goal_node(end_node);
                if(last_index != -1){
                    local_path = generate_final_course(last_index, end_node);
                    return local_path;
                }
            }
        }

        // Debug.Log("reached max iteration");

        last_index = search_best_goal_node(end_node);
            if(last_index != -1){
                local_path = generate_final_course(last_index, end_node);
                return local_path;
            }

        // Debug.Log("null");

        return null;
    }

    public Vector3[] generate_final_course(int goal_ind, RRTNode goal_node){
        List<Vector3> path = new List<Vector3>();
        path.Add(new Vector3(goal_node.x, 0.5f, goal_node.y));

        RRTNode current_node = node_list[goal_ind];

        while (current_node.parent != null)
        {
            path.Add(new Vector3(current_node.x, 0.5f, current_node.y));
            current_node = current_node.parent;
        }

        Vector3[] path_arr = path.ToArray();
        System.Array.Reverse(path_arr);
        return path_arr;
    }

    public int search_best_goal_node(RRTNode _goalpos){
        List<float> dist_to_goal_list = new List<float>();
        List<int> goal_inds = new List<int>();
        List<int> safe_goal_inds = new List<int>();
        List<float> costs = new List<float>();

        foreach(RRTNode _node in node_list){
            dist_to_goal_list.Add(calc_dist_to_goal(_node.x, _node.y, _goalpos));
        }

        foreach(float dist in dist_to_goal_list){
            if(dist <= expand_dis){
                goal_inds.Add(dist_to_goal_list.IndexOf(dist));
            }
        }

        foreach(int goal_ind in goal_inds){
            RRTNode t_node = steer(node_list[goal_ind], _goalpos, expand_dis);
            if(check_collision(t_node)){
                safe_goal_inds.Add(goal_ind);
            }
        }

        if(safe_goal_inds.Count == 0){
            return -1;
        }
        
        foreach(int idx in safe_goal_inds){
            costs.Add(node_list[idx].cost);
        }

        float min_cost = costs.Min();

        foreach(int idx in safe_goal_inds){
            if(node_list[idx].cost == min_cost){
                return idx;
            }
        }
        
        return -1;
    }

    public float calc_dist_to_goal(float x, float y, RRTNode _goalpos){
        float dx = x - _goalpos.x;
        float dy = y - _goalpos.y;

        return Mathf.Sqrt(Mathf.Pow(dx,2) + Mathf.Pow(dy,2));
    }



    public void rewire(RRTNode new_node, List<int> near_inds){
        foreach(int ind in near_inds){
            RRTNode near_node = node_list[ind];
            RRTNode edge_node = steer(new_node, near_node,expand_dis);
            if(edge_node == null){
                continue;
            }
            edge_node.cost = calc_new_cost(new_node,near_node);

            bool no_collision = check_collision(edge_node);
            bool improved_cost = near_node.cost > edge_node.cost;

            if(no_collision && improved_cost){
                near_node.x = edge_node.x;
                near_node.y = edge_node.y;
                near_node.cost = edge_node.cost;
                near_node.path_x = edge_node.path_x;
                near_node.path_y = edge_node.path_y;
                near_node.parent = edge_node.parent;
                propagate_cost_to_leaves(new_node);
            }
        }
    }

    public void propagate_cost_to_leaves(RRTNode parent_node){
        foreach(RRTNode node in node_list){
            if (node.parent == parent_node){
                node.cost = calc_new_cost(parent_node, node);
                propagate_cost_to_leaves(node);
            }
        }
    }

    public RRTNode choose_parent(RRTNode new_node, List<int> near_inds){
        if(near_inds.Count==0){
            return null;
        }

        List<float> costs = new List<float>();

        foreach(int i in near_inds){
            RRTNode near_node = node_list[i];
            RRTNode t_node = steer(near_node, new_node, expand_dis);
            if ((t_node!=null) && (check_collision(t_node))){
                costs.Add(calc_new_cost(near_node, new_node));
            } 
            else{
                costs.Add(float.MaxValue);
            }
        }

        float min_cost = costs.Min();

        if(min_cost == float.MaxValue){
            Debug.Log("There is no good path.");
            return null;
        }

        int min_ind = near_inds[costs.IndexOf(min_cost)];
        new_node = steer(node_list[min_ind], new_node, expand_dis);
        new_node.cost = min_cost;

        return new_node;
    }

    public float calc_new_cost(RRTNode from_node, RRTNode to_node){
        float distance;
        float angle;

        (distance, angle) = calc_distance_and_angle(from_node, to_node);
        return distance + from_node.cost;
    }

    public bool check_collision(RRTNode node){

        if(node == null){
            return false;
        }

        Vector3 start_pos = new Vector3(node.x, 0, node.y);
        if(Physics.CheckSphere(start_pos, robot_r, unwalkableMask)){
            if(Vector3.Distance(transform.position, start_pos) <= robot_r*2){
                return true;
            }
            return false;
        }

        return true;
    }

    public RRTNode steer(RRTNode from_node, RRTNode to_node, float extend_length){
        RRTNode newnode = new RRTNode(from_node.x, from_node.y);
        float distance;
        float angle;

        (distance, angle) = calc_distance_and_angle(from_node,to_node);
        
		
		newnode.path_x.Add(from_node.x);
        newnode.path_y.Add(from_node.y);
        
        if (extend_length > distance){
            extend_length  = distance;
        }

        int n_expand = (int)Mathf.Floor(extend_length/path_resolution);

        for(int i = 0; i < n_expand; i++){
            newnode.x += path_resolution * Mathf.Cos(angle);
            newnode.y += path_resolution * Mathf.Sin(angle);
            newnode.path_x.Add(newnode.x);
            newnode.path_y.Add(newnode.y);
        }

        (distance, angle) = calc_distance_and_angle(newnode, to_node);
        
        if(distance <= path_resolution){
            newnode.path_x.Add(to_node.x);
            newnode.path_y.Add(to_node.y);
            newnode.x = to_node.x;
            newnode.y = to_node.y;
        }

        newnode.parent = from_node;

        return newnode;
    }

    public (float, float) calc_distance_and_angle(RRTNode from_node, RRTNode to_node){
        float dx = to_node.x - from_node.x;
        float dy = to_node.y - from_node.y;

        float distance = Mathf.Sqrt(Mathf.Pow(dx,2) + Mathf.Pow(dy,2));
        float angle = Mathf.Atan2(dy,dx);

        return (distance, angle);
    }

    public int get_nearest_node_index(RRTNode randomNode){
        List<float> dist_arr = new List<float>();
        int min_ind;
        foreach(RRTNode _node in node_list){
            dist_arr.Add((Mathf.Pow(_node.x-randomNode.x,2)+(Mathf.Pow(_node.y-randomNode.y,2))));
        }
        min_ind = dist_arr.IndexOf(dist_arr.Min());
        return min_ind;
    }

    public List<int> find_near_nodes(RRTNode new_node){
        List<float> dist_arr = new List<float>();
        List<int> near_inds = new List<int>();
        int num_node = node_list.Count + 1;
        float r = connect_circle_dist * Mathf.Sqrt((Mathf.Log(num_node) / num_node));

        r = Mathf.Min(r, expand_dis);
        
        foreach(RRTNode _node in node_list){
            dist_arr.Add((Mathf.Pow(_node.x-new_node.x,2)+(Mathf.Pow(_node.y-new_node.y,2))));
        }

        foreach(float dist in dist_arr){
            if(dist <= Mathf.Pow(r,2)){
                near_inds.Add(dist_arr.IndexOf(dist));
            }
        }
        return near_inds;
    }


    public RRTNode get_random_node(RRTNode _goalpos){
        RRTNode randomNode;
        if(UnityEngine.Random.Range(0,100) > goal_sample_rate){
            randomNode = new RRTNode(UnityEngine.Random.Range(min_rand,max_rand), UnityEngine.Random.Range(min_rand,max_rand));
        }
        else{
            randomNode = new RRTNode(_goalpos.x, _goalpos.y);
        }
        
        return randomNode;
    }
}

public class RRTNode{
	public float x;
	public float y;
	public List<float> path_x = new List<float>();
	public List<float> path_y = new List<float>();
	public RRTNode parent = null;
    public float cost = 0.0f;


    public RRTNode(float _x, float _y) {
        x = _x;
        y = _y;
	}
}