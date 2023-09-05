import uvicorn

from src.app import app, get_generator_dummy
from src.generators import get_generator_dependency
from src.utils import get_logger, get_parser


logger = get_logger(__name__)


def main():
    args = get_parser().parse_args()

    # temporary solution for cli args passing
    generator_dependency = get_generator_dependency(args.model, args.device, args.tokenizer_checkpoint, args.assistant)
    app.dependency_overrides[get_generator_dummy] = generator_dependency

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
