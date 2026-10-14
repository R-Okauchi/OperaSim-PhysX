#if UNITY_EDITOR
using System;
using UnityEditor;
using UnityEditor.SceneManagement;

/// <summary>
/// エディタを開いたら指定のシーンで Play に入る (cap-pangaea scripts/cycle_video.py の録画: 人の操作なしで始める)。
/// 環境変数 CAP_UNITY_AUTOPLAY にシーンのパス (例 Assets/Scenes/SimpleScene.unity) があるときだけ、エディタの
/// セッションで 1 回だけ (Play に入るときのドメインの読み直しで繰り返さない)。無ければ何もしない。
/// </summary>
[InitializeOnLoad]
internal static class CapAutoPlay
{
    private const string DoneKey = "CapAutoPlay.done";

    static CapAutoPlay()
    {
        string scene = Environment.GetEnvironmentVariable("CAP_UNITY_AUTOPLAY");
        if (string.IsNullOrEmpty(scene) || SessionState.GetBool(DoneKey, false))
        {
            return;
        }
        EditorApplication.delayCall += () =>
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
            {
                return;
            }
            SessionState.SetBool(DoneKey, true);
            if (EditorSceneManager.GetActiveScene().path != scene)
            {
                EditorSceneManager.OpenScene(scene);
            }
            UnityEngine.Debug.Log($"[CapAutoPlay] {scene}: entering Play");
            EditorApplication.isPlaying = true;
        };
    }
}
#endif
