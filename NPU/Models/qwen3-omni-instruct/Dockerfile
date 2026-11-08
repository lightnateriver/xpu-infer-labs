FROM quay.io/ascend/vllm-ascend:v0.30.0
COPY patches /tmp/omni-patches
RUN cd /vllm-workspace/vllm && for name in modality input-alignment mrope-alignment; do git apply --check "/tmp/omni-patches/$name.patch" && git apply "/tmp/omni-patches/$name.patch" || exit 1; done
COPY serve.sh /usr/local/bin/serve-omni.sh
ENTRYPOINT ["bash", "/usr/local/bin/serve-omni.sh"]
