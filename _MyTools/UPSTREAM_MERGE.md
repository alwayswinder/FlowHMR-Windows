# FlowHMR 上游合并说明

本地扩展基于 `3fdc80c`。`export/` 保存 Character Studio BVH、BIP 与 Max 导出、SMPL-H 参考骨架计算和 MAXScript。`web/` 保存导出缓存及 Biped 预览绘制。`tests/` 保存骨架与手部校准回归。

| 文件 | 合并时保留的接入 |
| --- | --- |
| `flowhmr/core/export/smplh2fbx.py` | FBX 导出从 `_MyTools/export/smplh_rig.py` 复用关节命名和参考姿势。上游改动这些几何逻辑时同步更新共享实现，以保持 FBX 与 BVH 一致。 |
| `tools/web/server.py` | 增加 `has_bvh`、`has_bvh_max`、`has_bip` 能力字段。为 `/bvh`、`/bip`、`/bip-preview`、`/max` 接入动作、人选、体型参考与缓存逻辑，保留本机 Max 的访问检查。单独提供 Biped 预览 JS 路由。 |
| `tools/web/index.html`、`tools/web/static/js/api.js`、`tools/web/static/js/motion-export.js` | 增加两种 Biped 导出格式、首帧 T Pose 和 Max 场景选项。保留已有下载回调、多人选择与 ZIP 行为。 |
| `tools/web/static/js/viewer.js`、`tools/web/static/workspace.css`、`tools/web/static/js/locales/*.js` | 接入 BIP 回读骨架，复用现有时间轴和退出控件，保留新增选项的七种语言。 |
| `tools/windows/desktop_downloads.py` | 桌面下载识别 `.bvh`、`.bip`、`.max` 和多人 BVH/Max ZIP，沿用本机下载来源约束。 |
| `tests/test_motion_exports.py`、`.gitignore`、`.gitattributes`、`docs/retarget-export.md`、`THIRD_PARTY_NOTICES.md` | 保留路由回归、产物忽略规则、BsKeyTools 原始文件格式、导出说明和两项第三方授权。 |

合并后用 AutoGenerate 单视频验证 `GET /api/jobs/<job_id>/bvh`，并在本机网页用同一单人任务检查 BIP 预览和两条 Max 导出路线。用多人任务检查人选、起始帧和 ZIP。变更影响产物字节时更新 `web/exports.py` 或 `export/bip.py` 对应的缓存版本。

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_motion_exports _MyTools.tests.test_bvh_export
```
