Shader "Hidden/DepthToColor" 
{
    Properties {
        // 特になければ空でもOK
    }

    SubShader {
        Pass {
            ZTest Always
            ZWrite Off
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // カメラが自動的に生成する深度テクスチャ (DepthTextureMode.Depth が必要)
            sampler2D _CameraDepthTexture;

            float4 frag (v2f i) : SV_Target
            {
                // 0～1 の深度値: 最近距離が0、遠方が1
                float rawDepth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);

                // 線形深度にする場合は以下 (必要なら):
                // float linearDepth = LinearEyeDepth(rawDepth);

                // そのままグレースケールで返す
                return float4(rawDepth, rawDepth, rawDepth, 1.0);
            }
            ENDCG
        }
    }
    FallBack Off
}
