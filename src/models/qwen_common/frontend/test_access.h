#pragma once

#include "models/qwen_common/frontend/frontend.h"
#include "models/qwen_common/frontend/resources.h"
#include "models/qwen_common/frontend/prepared_prompt.h"

namespace infernix::models::qwen_common {

class FrontendTestAccess {
public:
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace infernix::models::qwen_common
