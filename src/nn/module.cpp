#include "module.h"

void module::train()
{
    training = true;
}

void module::eval()
{
    training = false;
}

bool module::is_training() const
{
    return training;
}