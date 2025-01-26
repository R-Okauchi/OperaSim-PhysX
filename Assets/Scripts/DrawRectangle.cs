using UnityEngine;

public class DrawRectangle : MonoBehaviour
{
    public Vector3[] corners = new Vector3[4];
    public Material lineMaterial;

    void Start()
    {
        // LineRendererを追加
        LineRenderer lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.positionCount = 5; // 4つの角 + 始点に戻る
        lineRenderer.loop = false;
        lineRenderer.useWorldSpace = false;

        // マテリアルと色を設定
        lineRenderer.material = lineMaterial;
        lineRenderer.startColor = Color.red;
        lineRenderer.endColor = Color.red;

        // 線の幅を設定
        lineRenderer.startWidth = 0.1f;
        lineRenderer.endWidth = 0.1f;

        // 四角形の頂点を設定
        corners[0] = new Vector3(-25, 0, -35);  // 左上
        corners[1] = new Vector3(25, 0, -35);   // 右上
        corners[2] = new Vector3(25, 0, 35);  // 右下
        corners[3] = new Vector3(-25, 0, 35); // 左下

        // 頂点情報をLineRendererに渡す
        for (int i = 0; i < 4; i++)
        {
            lineRenderer.SetPosition(i, corners[i]);
        }
        // 閉じるために最初の頂点を追加
        lineRenderer.SetPosition(4, corners[0]);
    }
}
