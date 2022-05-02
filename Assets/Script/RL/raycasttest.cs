using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;

public class raycasttest : MonoBehaviour
{
    
    public List<Ray> Rays = new List<Ray>();
    public List<Vector3> Rays_Vector = new List<Vector3>();
    public List<float> Ray_Dist = new List<float>();
    public RaycastHit hitData;
    public float maxRay = 20;
    public float robot_radius = 1;
    public LayerMask unwalkable;
    public Transform targetTrans;

    AGrid grid;

    Vector3[] globalpath;

    private Vector3 rnd_init_start;
    private Vector3 rnd_init_target;


    void Awake(){
        grid = GetComponent<AGrid>();
        
    }

    // Start is called before the first frame update
    void Start()
    {
        // for(int i = 0; i < 36; i++){
        //     test.Add(new Vector3(Mathf.Cos(Mathf.PI*i/10),0,Mathf.Sin(Mathf.PI*i/10)));
        // }
        GlobalPathRequestManager.RequestPath(transform.position,targetTrans.position, OnPathFound);
        if(globalpath!=null){
            for(int i = 0; i < globalpath.Length-1; i++){
            Debug.Log(globalpath[i]);
            }
        }
        
        
        Rays.Clear();
        Rays_Vector.Clear();
        Ray_Dist.Clear();

        for(int i = 0; i < 36; i++){
            Rays_Vector.Add(new Vector3(Mathf.Cos(Mathf.PI*i/10),0,Mathf.Sin(Mathf.PI*i/10)));
        }
    }

    public void OnPathFound(Vector3[] newPath, bool pathSuccessful) {
        Debug.Log(pathSuccessful);
		if (pathSuccessful) {
			globalpath = newPath;
		}
	}
    
    // Update is called once per frame
    void Update()
    {
        // for(int i = 0; i < 36; i++){
        //     Debug.DrawRay(transform.position,test[i] * 100,Color.red);
        // }
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

        if(Ray_Dist.Min() <= robot_radius){
            Debug.Log("collision!");
        }
    }

}
