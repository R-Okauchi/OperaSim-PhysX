#if UNITY_EDITOR
using System.Collections;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace CapKit.Tests
{
    /// <summary>
    /// A scene loaded afresh for a test. Upstream's soil keeps itself across scene loads (SoilParticleSettings puts its
    /// terrain under DontDestroyOnLoad and destroys any later one), so a second load's own terrain is destroyed at once
    /// - and with it what lives on it: the GNSS frame the scene's machines' sensors read, which then throw every frame,
    /// between tests too, where the next test fails on them (10-24: the dozer's two tests, then the soil's). The kept
    /// soil is destroyed first, so each test has its scene's own.
    /// </summary>
    internal static class CapTestScene
    {
        public static IEnumerator LoadFresh(string scene)
        {
            var kept = SoilParticleSettings.instance;
            if (kept != null)
            {
                Object.Destroy(kept.gameObject);
                SoilParticleSettings.instance = null;
                yield return null;
            }
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(scene, new LoadSceneParameters(LoadSceneMode.Single));
        }
    }
}
#endif
