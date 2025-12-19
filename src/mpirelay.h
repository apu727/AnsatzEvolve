#ifndef MPIRELAY_H
#define MPIRELAY_H

#include "serialise.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>



enum class MPICommand
{
    Shutdown = 0, // not actually in the map.
    Reply = 1, // Used to indicate a slave reply
    Broadcast = 2, // Used for broadcast messages. Not in the map, should never be received by that handler
    DirectSend = 3, // Direct send to Node.  Not in the map, should never be received by that handler
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

    std::string m_logPretext = "";

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
    bool IssueCommandToFreeNode(MPICommand comm, const serialDataContainer& data, std::function<void(char*)> callBack = std::function<void(char*)>());
    void waitForAll(); // waits for all nodes to be free again
    int getFreeNodeCount(){return m_freeNodes;}; // an estimate. If multiple people are queuing then there may be race conditions. If this becomes a problem, implement a semaphore type behaviour. Does not include the current node

    //These are sideband channels. A Broadcast assumes someone will receive.
    bool BroadcastToAllNodes(const serialDataContainer& data);
    serialDataContainer ReceiveBroadcast();
    serialDataContainer ReceiveFromNode(int nodeID);
    bool sendToNode(int nodeID,serialDataContainer& data);

    //Each node sends data to a lower node count and applies callBack to the data along with the received data
    /* On 8 nodes performs the operations:
     * 7->6 6: 6Data = callback(7Data,6Data)
     * 5->4 4: 4Data = callback(5Data,4Data)
     * 3->2 2: 2Data = callback(3Data,2Data)
     * 1->0 0: 0Data = callback(1Data,0Data)
     *
     * 6->4 4: 4Data = callback(6Data,4Data)
     * 2->0 0: 0Data = callback(2Data,0Data)
     *
     * 4->0 0: 0Data = callback(4Data,0Data)
     * return 0Data //Only Node 0
     * return nullptr // All other nodes
     */
    serialDataContainer treeReduceOp(const serialDataContainer& data, std::function<serialDataContainer (const serialDataContainer& theirs, const serialDataContainer& ours)> callBack);

    bool isMaster(){return m_amMaster;}
    void runSlaveLoop();
    int getRank(){return m_rank;}


};


#endif // MPIRELAY_H
