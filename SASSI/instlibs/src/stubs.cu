#include <sassi/sassi-core.hpp>
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////////////
///
///  This is a SASSI handler that handles only basic information about each
///  instrumented instruction.  The calls to this handler are placed by
///  convention *before* each instrumented instruction.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_before_handler(SASSIBeforeParams* bp)
{
}

