#pragma once

inline thread_local bool grad_enabled = true;

class no_grad_guard
{
private:
    bool previous_state;

public:
    no_grad_guard()
    : previous_state(grad_enabled){
        grad_enabled = false;
    }

    ~no_grad_guard(){
        grad_enabled = previous_state;
    }

    no_grad_guard(const no_grad_guard &) = delete;
    no_grad_guard &operator=(const no_grad_guard &) = delete;
};