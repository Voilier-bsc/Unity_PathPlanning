using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class agent_test : MonoBehaviour
{
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

    void Awake() {
        
    }

	void Start() {
		GlobalPathRequestManager.RequestPath(transform.position,target.position, OnPathFound);
	}

	public void OnPathFound(Vector3[] newPath, bool pathSuccessful) {
        Debug.Log(pathSuccessful);
		if (pathSuccessful) {
			path = newPath;
			targetIndex = 1;
			// StopCoroutine("FollowPath");
			// StartCoroutine("FollowPath");
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
	}
}
