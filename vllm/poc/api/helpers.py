import uuid

from fastapi import HTTPException, Request

from vllm.poc.api.models import PoCParamsModel


async def get_engine_client(request: Request):
    engine_client = getattr(request.app.state, "engine_client", None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


def check_params_match(request: Request, params: PoCParamsModel):
    """Check params match deployed config. Raises 409 on mismatch."""
    serving_models = getattr(request.app.state, "openai_serving_models", None)
    if serving_models and hasattr(serving_models, "base_model_paths"):
        base_paths = serving_models.base_model_paths
        if base_paths:
            model_path = base_paths[0].model_path
            served_names = [p.name for p in base_paths]
            valid_models = {model_path} | set(served_names)
            if params.model not in valid_models:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "params mismatch",
                        "requested": {
                            "model": params.model,
                            "seq_len": params.seq_len,
                            "k_dim": params.k_dim,
                        },
                        "deployed": {
                            "model": list(valid_models),
                            "seq_len": None,
                            "k_dim": None,
                        },
                    },
                )

    deployed = getattr(request.app.state, "poc_deployed", None)
    if deployed:
        mismatches = []
        if deployed.get("model") and params.model != deployed["model"]:
            mismatches.append("model")
        if deployed.get("seq_len") and params.seq_len != deployed["seq_len"]:
            mismatches.append("seq_len")
        if deployed.get("k_dim") and params.k_dim != deployed["k_dim"]:
            mismatches.append("k_dim")

        if mismatches:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "params mismatch",
                    "fields": mismatches,
                    "requested": {
                        "model": params.model,
                        "seq_len": params.seq_len,
                        "k_dim": params.k_dim,
                    },
                    "deployed": deployed,
                },
            )


def generate_request_id() -> str:
    return str(uuid.uuid4())
