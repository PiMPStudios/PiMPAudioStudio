# Third-Party Notices

PiMP Audio Studio is proprietary software, but it incorporates and interoperates
with third-party software, services, and model assets that remain governed by
their own licenses. The PiMP Audio Studio proprietary license does not replace
or restrict rights granted directly by those third-party licensors.

This file records the project's current direct software dependencies. It is a
development inventory, not yet the final release-bundle attribution report.
Before public distribution, the release process must collect the complete
licenses and required notices for direct and transitive dependencies actually
included in the shipped application.

## Direct Software Dependencies

| Component | Current project reference | License |
| --- | --- | --- |
| [MLX Swift](https://github.com/ml-explore/mlx-swift) | 0.31.3 | MIT, copyright 2023 ml-explore |
| [Swift Transformers](https://github.com/huggingface/swift-transformers) | 1.3.3 | Apache License 2.0 |
| [Speech Swift](https://github.com/soniqo/speech-swift) | revision `72a20dbb142d000b73d395b7bc62599fef8387e2` | Apache License 2.0 |
| [MCP Swift SDK](https://github.com/modelcontextprotocol/swift-sdk) | 0.12.1 | Apache License 2.0 for relicensed/new code; remaining historical code under MIT as described by that repository |
| [Demucs MLX Swift](Vendor/demucs-mlx-swift/LICENSE) | vendored source | MIT, copyright 2026 Kyle Howells |

The Swift packages above bring additional transitive dependencies. Their
license and NOTICE files must be preserved in the release attribution bundle.

## MIT License Notice: MLX Swift

Copyright (c) 2023 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## MIT License Notice: Demucs MLX Swift

Copyright (c) 2026 Kyle Howells

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The Apache License 2.0 text is available from the
[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0) and
must be included in distributions containing Apache-licensed components.

## Models, Datasets, and Remote Services

Large model weights are generally downloaded by the user and are not licensed
under the PiMP Audio Studio source license. ACE-Step, Stable Audio, annotation
models, language models, Hugging Face repositories, RunPod, and other remote
services or assets have separate terms. Users must obtain any required access,
accept applicable licenses, and comply with usage restrictions.

The portable remote worker may install or invoke Python, PyTorch, CUDA,
ACE-Step/Side-Step, PEFT, LyCORIS, and related packages on user-approved remote
hardware. Those components are not relicensed by PiMP Studios merely because
the worker automates their installation or use.
