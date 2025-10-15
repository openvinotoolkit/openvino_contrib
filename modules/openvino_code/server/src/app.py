from time import perf_counter
from typing import Dict, Optional, Union

from fastapi import Depends, FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, TypeAdapter

from src.generators import GeneratorFunctor
from src.utils import get_logger


logger = get_logger(__name__)


class GenerationParameters(BaseModel):
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    max_new_tokens: int = 60
    min_new_tokens: int = 0

    timeout: Optional[int] = None
    repetition_penalty: float = 1.0


class GenerationRequest(BaseModel):
    inputs: str
    parameters: GenerationParameters


class GenerationDocStringRequest(BaseModel):
    inputs: str = Field(
        ...,
        description="Function or Class body",
        example=(
            "def fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n"
            "        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
        ),
    )
    template: str = Field(
        ...,
        description=(
            "Doc string template with tab stops in format ${tab_stop_number:value[type | int | str | description]}"
        ),
        example=(
            '    """\n    ${1:}\n\n    Parameters\n    ----------\n    n : ${2:int}\n'
            "        ${3:[description]}\n\n    Returns\n    -------\n    ${4:[type]}\n"
            '        ${5:[description]}\n    """'
        ),
    )
    format: str = Field(
        ...,
        description="Doc string format passed from extension settings [google | numpy | sphinx | dockblockr | ...]",
        example="numpy",
    )
    definition: str = Field("", description="Function signature", example="def fibonacci(n):")
    parameters: GenerationParameters


class GenerationResponse(BaseModel):
    generated_text: str


app = FastAPI()


def get_generator_dummy():
    pass


@app.on_event("startup")
async def startup_event():
    # This print is a anchor for vs code extension to track that server is started
    SERVER_STARTED_STDOUT_ANCHOR = "OpenVINO Code Server started"
    logger.info(SERVER_STARTED_STDOUT_ANCHOR)


@app.get("/", include_in_schema=False)
def docs_redirect() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/api/health", status_code=200)
def health_check() -> Dict:
    return {}


@app.post("/api/generate", status_code=200, response_model=GenerationResponse)
async def generate(
    request: GenerationRequest,
    generator: GeneratorFunctor = Depends(get_generator_dummy),
) -> Dict[str, Union[int, str]]:
    logger.info(f"Request:\n{request}")

    start = perf_counter()
    generated_text: str = generator(request.inputs, request.parameters.model_dump())
    stop = perf_counter()

    if (elapsed := stop - start) > 1.5:
        logger.warning(f"Elapsed: {elapsed:.3f}s")
    else:
        logger.info(f"Elapsed: {elapsed:.3f}s")

    logger.info(f"Response:\n{generated_text}")
    return {"generated_text": generated_text}


@app.post("/api/generate_stream", status_code=200)
async def generate_stream(
    request: Request,
    generator: GeneratorFunctor = Depends(get_generator_dummy),
) -> StreamingResponse:
    generation_request = TypeAdapter(GenerationRequest).validate_python(await request.json())
    logger.info(generation_request)
    return StreamingResponse(
        generator.generate_stream(
            generation_request.inputs,
            generation_request.parameters.model_dump(),
            request,
        )
    )


@app.post("/api/summarize", status_code=200, response_model=GenerationResponse)
async def summarize(
    request: GenerationDocStringRequest,
    generator: GeneratorFunctor = Depends(get_generator_dummy),
):
    logger.info(request)

    start = perf_counter()
    generated_text: str = generator.summarize(
        request.inputs,
        request.template,
        request.definition,
        request.format,
        request.parameters.model_dump(),
    )
    stop = perf_counter()

    if (elapsed := stop - start) > 1.5:
        logger.warning(f"Elapsed: {elapsed:.3f}s")
    else:
        logger.info(f"Elapsed: {elapsed:.3f}s")

    logger.info(f"Response: {generated_text}")
    return {"generated_text": generated_text}


@app.post("/api/summarize_stream", status_code=200)
async def summarize_stream(
    request: GenerationDocStringRequest,
    generator: GeneratorFunctor = Depends(get_generator_dummy),
) -> StreamingResponse:
    logger.info(request)
    return StreamingResponse(
        generator.summarize_stream(
            request.inputs,
            request.template,
            request.definition,
            request.format,
            request.parameters.model_dump(),
        )
    )
