from utils import (
    build_openai_messages,
    build_openai_request_kwargs,
    build_parser,
    create_openai_client,
    get_api_key,
    get_image_data_url,
    run_evaluation,
)


BACKEND_NAME = "qwen3-vl"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8888/v1"


class Qwen3VLAdapter:
    def __init__(self, args):
        if not args.model:
            raise ValueError("Missing model name. Provide --model for the qwen3-vl backend.")
        self.args = args
        self.client = create_openai_client(
            base_url=args.base_url or DEFAULT_BASE_URL,
            api_key=get_api_key(args, default="EMPTY"),
        )

    def generate(self, example, prompt, context):
        messages = build_openai_messages(
            prompt=prompt,
            image_data_url=get_image_data_url(example, context),
            system_prompt=self.args.system_prompt,
        )
        request_kwargs = build_openai_request_kwargs(self.args)
        request_kwargs["messages"] = messages
        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""


ADAPTER_CLASS = Qwen3VLAdapter


def run(args):
    run_evaluation(args, backend_name=BACKEND_NAME, adapter_factory=Qwen3VLAdapter)


def parse_args():
    parser = build_parser("Evaluate EndoBench with a Qwen3-VL model.", default_model=DEFAULT_MODEL)
    parser.set_defaults(base_url=DEFAULT_BASE_URL, api_key="EMPTY")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
