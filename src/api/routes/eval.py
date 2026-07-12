"""Routes for running the golden-dataset eval harness and reading results."""

from pydantic import BaseModel

from fastapi import APIRouter

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRunRequest(BaseModel):
    dataset_name: str


@router.post("/run")
async def run_eval(request: EvalRunRequest):
    """Run the eval harness over the named golden dataset and return a run id."""
    raise NotImplementedError


@router.get("/runs/{run_id}")
async def get_eval_run(run_id: str):
    """Return the stored results for a previously run eval run."""
    raise NotImplementedError
