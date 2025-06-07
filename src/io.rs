// ========================================================================================
//                   THE PHYSICAL-TO-LOGICAL ABSTRACTION LAYER
// ========================================================================================
// This module is the sole point of contact between the clean, logical world of our
// engine and the messy, physical reality of the filesystem. It is an airtight
// abstraction boundary.
//
//      The rest of the application does not know what a
//     "magic number" is, how genotypes are packed, or what a byte offset means.
