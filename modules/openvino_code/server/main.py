from src.utils import get_parser, setup_logger


# Logger should be set up before other imports to propagate logging config to other packages
setup_logger()

import uvicorn  # noqa: E402

from src.app import app, get_generator_dummy  # noqa: E402
from src.generators import get_generator_dependency  # noqa: E402


def main():
    args = get_parser().parse_args()

    # temporary solution for cli args passing
    generator_dependency = get_generator_dependency(args.model, args.device, args.tokenizer_checkpoint, args.assistant)
    app.dependency_overrides[get_generator_dummy] = generator_dependency

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
