﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// ZX200のシミュレーションモデルに対して細かな干渉検出設定を行う
/// </summary>
public class CustomCollisionZX200 : MonoBehaviour
{
    Transform arm_link;
    Transform bucket_inner_link;

    // Start is called before the first frame update
    void Start()
    {
        arm_link = gameObject.transform.Find("base_link/body_link/boom_link/arm_link/Collisions");
        bucket_inner_link = gameObject.transform.Find("base_link/body_link/boom_link/arm_link/bucket_link/bucket_inner");
        
        foreach (Collider c1 in arm_link.GetComponentsInChildren(typeof(Collider)))
        {
            foreach (Collider c2 in bucket_inner_link.GetComponentsInChildren(typeof(Collider)))
            {
                Physics.IgnoreCollision(c1, c2);
            }
        }
    }
}
