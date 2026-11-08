# 自备冻结输入

输入 JSON：`{"warmup": {"messages": [...]}, "samples": [{"messages": [...]}, ...15条...], "schedule": [0,1,...共56个索引...]}`。

每条 `messages` 是原始 OpenAI chat messages，包含调用方 system prompt、参考图和视频。示例媒体项：`{"type":"video_url","video_url":{"url":"/absolute/path/video.mp4"}}`；参考图用 image_url。客户端读本地路径转 data URL，也接受预先准备的 data URL。禁止 HTTP URL，以便保证输入本地冻结。每项可以提供 `input_tokens`，但运行仍会 tokenize 并校验实际值。

`warmup` 必须使用独立媒体与提示，不能出现在正式 samples 中。`schedule` 必须覆盖所有 15 个索引，重复次数分别 3 或 4，总长 56；按自己的冻结清单记录顺序，不提供虚构历史清单。路径、媒体及 prompts 不提交 Git。

若需业务 schema 校验，提供 Python 文件，定义 `validate(output_text) -> bool`；通过 `--validator /path/to/validator.py` 启用。默认只验证 JSON，不等于原业务 schema 或语义正确。
