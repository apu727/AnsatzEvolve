#ifndef MPIRELAY_H
#define MPIRELAY_H

#include "serialise.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <map>



enum class MPICommand
{
    Shutdown = 0, // not actually in the map.
    Reply = 1, // Used to indicate a slave reply
    //Reserved for future stuff
    //32-64 Hamiltonian stuff
    HamApplyToVector = 33,
    // 64 Onwards ??
    //RDM stuff?
    // 96 Onwards Statevector stuff??
};


class MPIRelay
{
    int m_totalNodes = 0; // number of nodes that arent this one.
    int m_freeNodes = 0;
    int m_rank = 0;
    bool m_amMaster = false;

    std::map<int,std::function<void(char*)>> m_nodeCallBackMap;
    std::map<MPICommand,serialDataContainer (*)(char*,size_t)> m_registeredMPICommands; // association between methods to call on the remote and function pointers.

    //Slave data
    bool m_isRunning = false;

    MPIRelay();
    ~MPIRelay();
    void registerCommands();
public:
    static MPIRelay& getInstance()
    {
        static MPIRelay me;
        return me;
    }
    // Issue to a free node. Returns false if no free nodes available
    // Comm is the command, Data is the data to send, callBack is called when data is returned.
    // This function must be called from a single thread, wait for all will wait for alll jobs. So if jobs are being queued you may be stuck a while. TODO futures.
    // Data must remain valid until the end of the function call.
    // The callback must deal with the data before returning.
    bool IssueCommandToFreeNode(MPICommand comm, const serialDataContainer& data, std::function<void(char*)> callBack);
    void waitForAll(); // waits for all nodes to be free again
    int getFreeNodeCount(){return m_freeNodes;}; // an estimate. If multiple people are queuing then there may be race conditions. If this becomes a problem, implement a semaphore type behaviour. Does not include the current node

    bool isMaster(){return m_amMaster;};
    void runSlaveLoop();


};


#endif // MPIRELAY_H
