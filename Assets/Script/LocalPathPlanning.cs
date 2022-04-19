using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class LocalPathPlanning : MonoBehaviour
{
    public int max_iter; //500
    public int goal_sample_rate; //5
    public int min_rand;
    public int max_rand;
    public float expand_dis; // 3.0
    public float path_resolution; //0.5
    public float connect_circle_dist; //50.0
    public LayerMask unwalkableMask;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void localplanning(RRTNode _startpos, RRTNode _goalpos){
        List<RRTNode> node_list = new List<RRTNode>();
        node_list.Add(_startpos);
        RRTNode rnd_node;
        int nearest_ind;
        RRTNode nearest_node;
        

        Vector3[] local_path;




        for(int i = 0; i < max_iter; i++){
            rnd_node = get_random_node(_goalpos);
            nearest_ind = get_nearest_node_index(node_list, rnd_node);
            RRTNode New_Node = steer(node_list[nearest_ind], rnd_node, expand_dis);
            nearest_node = node_list[nearest_ind];

            New_Node.cost = nearest_node.cost + (Mathf.Pow(New_Node.x-nearest_node.x,2)+(Mathf.Pow(New_Node.y-nearest_node.y,2)));
            
            if (check_collision(New_Node)){
                List<int> near_inds = find_near_nodes(node_list, New_Node);
                RRTNode node_with_updated_parent = choose_parent(node_list, New_Node, near_inds);
                if(node_with_updated_parent != null){
                    //rewire
                    node_list.Add(node_with_updated_parent);
                }
                else{
                    node_list.Add(New_Node);
                }

            }

        }
    }

    public void rewire(){

    }

    RRTNode choose_parent(List<RRTNode> _nodeList, RRTNode new_node, List<int> near_inds){
        if(near_inds.Count==0){
            return null;
        }

        List<float> costs = new List<float>();

        foreach(int i in near_inds){
            RRTNode near_node = _nodeList[i];
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
        new_node = steer(_nodeList[min_ind], new_node, expand_dis);
        new_node.cost = min_cost;

        return new_node;
    }

    float calc_new_cost(RRTNode from_node, RRTNode to_node){
        float distance;
        float angle;

        (distance, angle) = calc_distance_and_angle(from_node, to_node);
        return distance + from_node.cost;
    }

    bool check_collision(RRTNode node){

        if(node == null){
            return false;
        }

        Vector3 start_pos = new Vector3(node.x, 0, node.y);
        if(Physics.CheckSphere(start_pos, 0.1f, unwalkableMask)){
            return false;
        }

        return true;
    }

    RRTNode steer(RRTNode from_node, RRTNode to_node, float extend_length){
        RRTNode newnode = new RRTNode(from_node.x, from_node.y);
        float distance;
        float angle;

        (distance, angle) = calc_distance_and_angle(from_node,to_node);
        newnode.path_x.Add(newnode.x);
        newnode.path_y.Add(newnode.y);
        
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

    (float, float) calc_distance_and_angle(RRTNode from_node, RRTNode to_node){
        float dx = from_node.x - to_node.x;
        float dy = from_node.y - to_node.y;

        float distance = Mathf.Sqrt(Mathf.Pow(dx,2) + Mathf.Pow(dy,2));
        float angle = Mathf.Atan2(dy,dx);

        return (distance, angle);
    }

    int get_nearest_node_index(List<RRTNode> _nodeList, RRTNode randomNode){
        List<float> dist_arr = new List<float>();
        int min_ind;
        foreach(RRTNode _node in _nodeList){
            dist_arr.Add((Mathf.Pow(_node.x-randomNode.x,2)+(Mathf.Pow(_node.y-randomNode.y,2))));
        }
        min_ind = dist_arr.IndexOf(dist_arr.Min());
        return min_ind;
    }

    List<int> find_near_nodes(List<RRTNode> _nodeList, RRTNode new_node){
        List<float> dist_arr = new List<float>();
        List<int> near_inds = new List<int>();
        int num_node = _nodeList.Count + 1;
        float r = connect_circle_dist * Mathf.Sqrt((Mathf.Log(num_node) / num_node));

        r = Mathf.Min(r, expand_dis);
        
        foreach(RRTNode _node in _nodeList){
            dist_arr.Add((Mathf.Pow(_node.x-new_node.x,2)+(Mathf.Pow(_node.y-new_node.y,2))));
        }

        foreach(float dist in dist_arr){
            if(dist <= Mathf.Pow(r,2)){
                near_inds.Add(dist_arr.IndexOf(dist));
            }
        }
        return near_inds;
    }


    RRTNode get_random_node(RRTNode _goalpos){
        RRTNode randomNode;
        if(Random.Range(0,100) > goal_sample_rate){
            randomNode = new RRTNode(Random.Range(min_rand,max_rand), Random.Range(min_rand,max_rand));
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
	public List<float> path_x;
	public List<float> path_y;
	public RRTNode parent = null;
    public float cost = 0.0f;


    public RRTNode(float _x, float _y) {
        x = _x;
        y = _y;
	}
}