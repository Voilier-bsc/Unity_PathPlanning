using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Unit : MonoBehaviour {
	public Transform target;
	float speed = 1000;
	Vector3[] path;
	int targetIndex;
    public LayerMask unwalkableMask;
    private Vector3 rnd_init_start;
    private Vector3 rnd_init_target;
    AGrid grid;


    void Awake() {
        grid = GetComponent<AGrid>();
		while(true){
			rnd_init_start = new Vector3(Random.Range(-4800,4800),transform.position.y,Random.Range(-4800,4800));
			rnd_init_target = new Vector3(Random.Range(-4800,4800),target.position.y,Random.Range(-4800,4800));
			if((!Physics.CheckSphere(rnd_init_start, 200, unwalkableMask))&& (!Physics.CheckSphere(rnd_init_target, 200, unwalkableMask)) && (Vector3.Distance(rnd_init_start, rnd_init_target) > 300f)){
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
			targetIndex = 0;
			StopCoroutine("FollowPath");
			StartCoroutine("FollowPath");
		}
	}

	IEnumerator FollowPath() {
		Vector3 currentWaypoint = path[0];
		while (true) {
			if (transform.position == currentWaypoint) {
				targetIndex ++;
				if (targetIndex >= path.Length) {
					yield break;
				}
				currentWaypoint = path[targetIndex];
			}
			// local path planning 진행

			transform.position = Vector3.MoveTowards(transform.position,currentWaypoint,speed * Time.deltaTime);
			yield return null;

		}
	}

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

