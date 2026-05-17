using UnityEngine;

/// <summary>
/// LiDARPublisherのRaycastヒットポイントをParticleSystemで点群として可視化する。
/// LiDARPublisherと同じGameObjectにアタッチする。
/// </summary>
[RequireComponent(typeof(LiDARPublisher))]
public class LiDARVisualizer : MonoBehaviour
{
    [Header("Visualization Settings")]
    [Tooltip("可視化の有効/無効")]
    public bool enableVisualization = true;

    [Tooltip("パーティクルのサイズ")]
    public float particleSize = 0.15f;

    [Tooltip("近距離の色 (intensity=1.0)")]
    public Color nearColor = Color.green;

    [Tooltip("遠距離の色 (intensity=0.0)")]
    public Color farColor = Color.red;

    [Tooltip("パーティクルの表示時間 (秒)。LiDAR更新間隔に合わせる")]
    public float particleLifetime = 0.15f;

    private LiDARPublisher lidarPublisher;
    private ParticleSystem ps;
    private ParticleSystem.Particle[] particles;

    void Start()
    {
        lidarPublisher = GetComponent<LiDARPublisher>();

        // ParticleSystemをコードで構築
        ps = gameObject.AddComponent<ParticleSystem>();

        var main = ps.main;
        main.loop = false;
        main.playOnAwake = false;
        main.startLifetime = particleLifetime;
        main.startSpeed = 0f;
        main.startSize = particleSize;
        main.simulationSpace = ParticleSystemSimulationSpace.World;
        main.maxParticles = lidarPublisher.horizontalRays * lidarPublisher.verticalChannels;

        var emission = ps.emission;
        emission.enabled = false;

        var shape = ps.shape;
        shape.enabled = false;

        // デフォルトRendererの設定
        var renderer = ps.GetComponent<ParticleSystemRenderer>();
        renderer.renderMode = ParticleSystemRenderMode.Billboard;
        renderer.material = new Material(Shader.Find("Particles/Standard Unlit"));
        renderer.material.SetFloat("_Mode", 0); // Additive blending off

        int maxParticles = lidarPublisher.horizontalRays * lidarPublisher.verticalChannels;
        particles = new ParticleSystem.Particle[maxParticles];
    }

    void LateUpdate()
    {
        if (!enableVisualization || lidarPublisher == null || lidarPublisher.hitCount == 0)
        {
            if (ps != null && ps.particleCount > 0)
                ps.Clear();
            return;
        }

        int count = lidarPublisher.hitCount;

        // パーティクル配列のサイズを確認
        if (particles == null || particles.Length < count)
            particles = new ParticleSystem.Particle[count];

        for (int i = 0; i < count; i++)
        {
            particles[i].position = lidarPublisher.hitPoints[i];
            particles[i].startSize = particleSize;
            particles[i].remainingLifetime = particleLifetime;
            particles[i].startLifetime = particleLifetime;

            // 距離に基づく色補間: intensity=1(近い)→nearColor, intensity=0(遠い)→farColor
            float t = lidarPublisher.hitIntensities[i];
            particles[i].startColor = Color.Lerp(farColor, nearColor, t);
        }

        ps.SetParticles(particles, count);
    }

    void OnValidate()
    {
        // エディタで値を変更した場合にParticleSystemのパラメータを反映
        if (ps != null)
        {
            var main = ps.main;
            main.startSize = particleSize;
            main.startLifetime = particleLifetime;
        }
    }
}
