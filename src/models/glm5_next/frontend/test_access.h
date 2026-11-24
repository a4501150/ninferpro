#pragma once

#include "models/glm5_next/frontend/frontend.h"
#include "models/glm5_next/frontend/resources.h"
#include "models/glm5_next/frontend/prepared_prompt.h"

namespace infernix::models::glm5_next {

class FrontendTestAccess {
public:
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace infernix::models::glm5_next
