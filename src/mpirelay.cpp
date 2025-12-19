#include "mpirelay.h"
#include "logger.h"

#include "hamiltonianmatrix.h"

#ifdef USEMPI
#include <mpi.h>
#endif
constexpr bool traceMessages = false;
constexpr int maxPayloadSize = 2000000000;

struct payloadHeader
{
    MPICommand command;
    int64_t payloadSize;
    size_t alignment; // alignment needed of the resultant payload
};

MPIRelay::MPIRelay()
{
#ifdef USEMPI
    int size,provided;
    MPI_Init_thread(NULL,NULL,MPI_THREAD_FUNNELED,&provided);
    releaseAssert(provided == MPI_THREAD_FUNNELED,"provided == MPI_THREAD_FUNNELED");
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    m_logPretext = std::string("Node") + std::to_string(m_rank) + "> ";
    m_totalNodes = size - 1;
    m_freeNodes = m_totalNodes;

    m_amMaster = m_rank == 0;
    if (m_amMaster)
    {
        registerCommands();
        logger().log(m_logPretext + "Master node initialized with workers", m_totalNodes);

    }
    registerCommands();
#else
    releaseAssert(false,"MPIRelay called without MPI build");
#endif
}

MPIRelay::~MPIRelay()
{
#ifdef USEMPI
    if (m_amMaster)
    {
        waitForAll();
        int cmdInt = static_cast<int>(MPICommand::Shutdown);
        for (int i = 1; i < m_totalNodes+1; i++)
        {
            logger().log(m_logPretext + "Sending Shutdown", i);
            MPI_Send(&cmdInt, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
#else
    releaseAssert(false,"~MPIRelay called without MPI build");
#endif
}

void MPIRelay::registerCommands()
{
    m_registeredMPICommands[MPICommand::HamApplyToVector] = MPICOMMAND_HamApplyToVector;
}

bool MPIRelay::IssueCommandToFreeNode(MPICommand comm, const serialDataContainer& data, std::function<void(char*)> callBack)
{
#ifdef USEMPI
    if (m_freeNodes <= 0 || !m_amMaster || m_totalNodes == 0)
        return false;

    int targetNode = -1;
    for (int node = 1; node < m_totalNodes+1; ++node)
    {
        if (m_nodeCallBackMap.find(node) == m_nodeCallBackMap.end())
        {
            targetNode = node;
            break;
        }
    }
    if (targetNode == -1)
        return false;

    m_nodeCallBackMap[targetNode] = callBack;
    --m_freeNodes;
    payloadHeader payload;
    payload.command = comm;
    payload.payloadSize = data.ptr ? data.size : 0;
    // Determine what it is aligned to here and replicate on the other side.
    // Ideally we actually want to be told this as a memory can accidentally have higher alignment than needed.
    // This leads to uneccessary reallocations on the other side
    payload.alignment = data.alignment;


    if (traceMessages) logger().log(m_logPretext + "Sending command", static_cast<int>(payload.command));
    if (traceMessages) logger().log(m_logPretext + "Sending payload.payloadSize", payload.payloadSize);
    if (traceMessages) logger().log(m_logPretext + "Sending payload.alignment", payload.alignment);

    MPI_Send(&payload, sizeof(payloadHeader), MPI_CHAR, targetNode, 0, MPI_COMM_WORLD);
    int64_t sentBytes = 0;
    while (data.ptr && sentBytes < payload.payloadSize)
    {
        int toSend = std::min(payload.payloadSize - sentBytes,(int64_t)maxPayloadSize);
        MPI_Send(data.ptr.get()+sentBytes, toSend, MPI_CHAR, targetNode, 0, MPI_COMM_WORLD);
        sentBytes += toSend;
    }

    return true;
#else
    releaseAssert(false,"IssueCommandToFreeNode called without MPI build");
    return false;
#endif
}

//This is probably a little broken. We should really give back futures like we do for threadpool.
void MPIRelay::waitForAll()
{
#ifdef USEMPI
    if (!m_amMaster)
    {
        logger().log(m_logPretext + "Slave called waitForAll");
        return;
    }

    MPI_Status status;
    char* dataBuffer = nullptr;
    size_t bufferSize = 0;
    size_t bufferAlignment = 1;

    if (traceMessages) logger().log(m_logPretext + "Enter Waiting for Nodes", m_totalNodes - m_freeNodes);
    while (m_freeNodes != m_totalNodes)
    {
        if (traceMessages) logger().log(m_logPretext + "Waiting for Nodes", m_totalNodes - m_freeNodes);
        // Wait for any worker to return
        MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int node = status.MPI_SOURCE;
        if (traceMessages) logger().log(m_logPretext + "Data from node", node);
        // Determine payload size
        payloadHeader payload;
        MPI_Recv(&payload, sizeof(payload), MPI_CHAR, node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (traceMessages) logger().log(m_logPretext + "Command", static_cast<int>(payload.command));
        if (traceMessages) logger().log(m_logPretext + "payloadSize", payload.payloadSize);

        if (payload.payloadSize != 0)
        {

            if (bufferSize < static_cast<size_t>(payload.payloadSize) || bufferAlignment < payload.alignment)
            {
                if (dataBuffer)
                    operator delete[] (dataBuffer,std::align_val_t(bufferAlignment));
                dataBuffer = new (std::align_val_t(payload.alignment)) char[payload.payloadSize];
                bufferAlignment = payload.alignment;
            }

            // Receive payload
            int64_t bytesReceived = 0;
            while (bytesReceived < payload.payloadSize)
            {
                int toRead;
                MPI_Probe(node, 0, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status,MPI_CHAR,&toRead);
                if (traceMessages) logger().log(m_logPretext + "Reading", toRead);
                releaseAssert(toRead + bytesReceived <= payload.payloadSize,"toRead + bytesReceived <= payload.payloadSize");

                MPI_Recv(dataBuffer+bytesReceived, toRead, MPI_CHAR, node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bytesReceived += toRead;
            }
            releaseAssert(bytesReceived == payload.payloadSize,"bytesReceived == payload.payloadSize");
            if (traceMessages) logger().log(m_logPretext + "Done Read");
        }
        // --- Pop callback for this node ---
        std::function<void(char*)> cb;
        cb = std::move(m_nodeCallBackMap[node]);
        m_nodeCallBackMap.erase(node);
        ++m_freeNodes;

        // Execute returned callback
        if (cb)
            cb(dataBuffer);
        if (traceMessages) logger().log(m_logPretext + "Done CallBack");
    }
    if (dataBuffer)
        operator delete[] (dataBuffer,std::align_val_t(bufferAlignment));
#else
    releaseAssert(false,"waitForAll called without MPI build");
#endif
}

bool MPIRelay::BroadcastToAllNodes(const serialDataContainer &data)
{
    releaseAssert(m_freeNodes == 0,m_logPretext + "BroadCastToAllNodes but not all nodes are waiting");
    releaseAssert(m_rank == 0,"Can only broadcast from rank 0");

    payloadHeader payload;
    payload.command = MPICommand::Broadcast;
    payload.payloadSize = data.size;
    payload.alignment = data.alignment;


    if (traceMessages) logger().log(m_logPretext + "Broadcasting command", static_cast<int>(payload.command));
    if (traceMessages) logger().log(m_logPretext + "Broadcasting payload.payloadSize", payload.payloadSize);
    if (traceMessages) logger().log(m_logPretext + "Broadcasting payload.alignment", payload.alignment);

    MPI_Bcast(&payload, sizeof(payloadHeader), MPI_CHAR, 0, MPI_COMM_WORLD);
    int64_t sentBytes = 0;
    while (sentBytes < payload.payloadSize)
    {
        int toSend = std::min(payload.payloadSize - sentBytes,(int64_t)maxPayloadSize);
        MPI_Bcast(data.ptr.get()+sentBytes, toSend, MPI_CHAR, 0, MPI_COMM_WORLD);
        sentBytes += toSend;
    }


    return true;
}

serialDataContainer MPIRelay::ReceiveBroadcast()
{
    serialDataContainer ret;

    payloadHeader payload;
    MPI_Bcast(&payload, sizeof(payload), MPI_CHAR, 0, MPI_COMM_WORLD);

    releaseAssert(payload.command == MPICommand::Broadcast, "ReceiveBroadcast not a Broadcast");

    if (traceMessages) logger().log(m_logPretext + "Received Broadcast command", static_cast<int>(payload.command));
    if (traceMessages) logger().log(m_logPretext + "Broadcast PayloadSize", static_cast<int>(payload.payloadSize));
    if (traceMessages) logger().log(m_logPretext + "Broadcast Payload alignment", static_cast<int>(payload.alignment));

    ret.ptr = std::shared_ptr<char[]>(new (std::align_val_t(payload.alignment)) char[payload.payloadSize],[al = payload.alignment](char* p){operator delete[] (p,std::align_val_t(al));});
    ret.size = payload.payloadSize;
    ret.alignment = payload.alignment;

    int64_t bytesReceived = 0;
    while (bytesReceived < payload.payloadSize)
    {
        int toRead = std::min(payload.payloadSize - bytesReceived,(int64_t)maxPayloadSize);
        if (traceMessages)  logger().log(m_logPretext + "reading Broadcast ", toRead);
        releaseAssert(toRead + bytesReceived <= payload.payloadSize,"toRead + bytesReceived <= payload.payloadSize");

        MPI_Bcast(ret.ptr.get() + bytesReceived, toRead, MPI_CHAR, 0, MPI_COMM_WORLD);
        bytesReceived += toRead;
    }
    releaseAssert(bytesReceived == payload.payloadSize,m_logPretext + "bytesReceived == payload.payloadSize");
    if (traceMessages) logger().log(m_logPretext + "Done ReceiveBroadcast");
    return ret;

}

serialDataContainer MPIRelay::ReceiveFromNode(int nodeID)
{
    releaseAssert(nodeID < m_totalNodes+1,m_logPretext + "Invalid NodeID given: " + std::to_string(nodeID));
    MPI_Status status;
    serialDataContainer ret;

    payloadHeader payload;
    MPI_Recv(&payload, sizeof(payload), MPI_CHAR, nodeID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    releaseAssert(payload.command != MPICommand::Shutdown, m_logPretext + "Shutdown Received by ReceiveFromNode");
    releaseAssert(payload.command == MPICommand::DirectSend, "ReceiveFromNode Invalid command metadata");

    if (traceMessages) logger().log(m_logPretext + "Received command", static_cast<int>(payload.command));
    if (traceMessages) logger().log(m_logPretext + "PayloadSize", static_cast<int>(payload.payloadSize));
    if (traceMessages) logger().log(m_logPretext + "Payload alignment", static_cast<int>(payload.alignment));

    ret.ptr = std::shared_ptr<char[]>(new (std::align_val_t(payload.alignment)) char[payload.payloadSize],[al = payload.alignment](char* p){operator delete[] (p,std::align_val_t(al));});
    ret.size = payload.payloadSize;
    ret.alignment = payload.alignment;

    int64_t bytesReceived = 0;
    while (bytesReceived < payload.payloadSize)
    {
        int toRead;
        MPI_Probe(nodeID, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status,MPI_CHAR,&toRead);
        if (traceMessages)  logger().log(m_logPretext + "reading ", toRead);
        releaseAssert(toRead + bytesReceived <= payload.payloadSize,"toRead + bytesReceived <= payload.payloadSize");

        MPI_Recv(ret.ptr.get() + bytesReceived, toRead, MPI_CHAR, nodeID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        bytesReceived += toRead;
    }
    releaseAssert(bytesReceived == payload.payloadSize,m_logPretext + "bytesReceived == payload.payloadSize");
    if (traceMessages) logger().log(m_logPretext + "Done ReceiveFromNode");
    return ret;
}

bool MPIRelay::sendToNode(int nodeID, serialDataContainer &data)
{
    payloadHeader payload;
    payload.command = MPICommand::DirectSend;
    payload.payloadSize = data.size;
    payload.alignment = data.alignment;


    if (traceMessages) logger().log(m_logPretext + "Sending command", static_cast<int>(payload.command));
    if (traceMessages) logger().log(m_logPretext + "Sending payload.payloadSize", payload.payloadSize);
    if (traceMessages) logger().log(m_logPretext + "Sending payload.alignment", payload.alignment);

    MPI_Send(&payload, sizeof(payloadHeader), MPI_CHAR, nodeID, 0, MPI_COMM_WORLD);
    int64_t sentBytes = 0;
    while (sentBytes < payload.payloadSize)
    {
        int toSend = std::min(payload.payloadSize - sentBytes,(int64_t)maxPayloadSize);
        MPI_Send(data.ptr.get()+sentBytes, toSend, MPI_CHAR, nodeID, 0, MPI_COMM_WORLD);
        sentBytes += toSend;
    }

    return true;
}

serialDataContainer MPIRelay::treeReduceOp(const serialDataContainer &data, std::function<serialDataContainer (const serialDataContainer& theirs, const serialDataContainer& ours)> callBack)
{
    //Each node sends data to a lower node count and applies callBack to the data along with the received data
    /* On 8 nodes performs the operations:
     * 7->6 6: 6Data = callback(7Data,6Data) 0b111 -> 0b110
     * 5->4 4: 4Data = callback(5Data,4Data) 0b101 -> 0b100
     * 3->2 2: 2Data = callback(3Data,2Data) 0b011 -> 0b010
     * 1->0 0: 0Data = callback(1Data,0Data) 0b001 -> 0b000
     *
     * 6->4 4: 4Data = callback(6Data,4Data) 0b110 -> 0b100
     * 2->0 0: 0Data = callback(2Data,0Data) 0b010 -> 0b000
     *
     * 4->0 0: 0Data = callback(4Data,0Data) 0b100 -> 0b000
     * return 0Data //Only Node 0
     * return nullptr // All other nodes
     */

    serialDataContainer currentData = data;
    for (int treeDepth = 0; 1<<treeDepth < m_totalNodes +1; ++treeDepth)
    {
        int sender = m_rank | 1<<treeDepth;
        int sendee = sender ^ (1<<treeDepth);
        std::string opString = " " + std::to_string(sender) + " -> " + std::to_string(sendee) + " ";
        if (sender >= m_totalNodes+1)
        {
            if (traceMessages) logger().log(m_logPretext + "treeReduceOp>" + opString + "sender doesnt exist, skipping");
            continue;
        }
        if (sender == m_rank)
        {
            if (traceMessages) logger().log(m_logPretext + "treeReduceOp>" + opString + "sending");
            sendToNode(sendee,currentData);
            if (traceMessages) logger().log(m_logPretext + "treeReduceOp>" + opString + "returning");
            return serialDataContainer();
        }
        else
        {
            if (traceMessages) logger().log(m_logPretext + "treeReduceOp>" + opString + "receiving");
            serialDataContainer theirs = ReceiveFromNode(sender);
            if (traceMessages) logger().log(m_logPretext + "treeReduceOp>" + opString + "received");
            currentData = callBack(theirs,currentData);
            if (traceMessages) logger().log(m_logPretext + "treeReduceOp>" + opString + "done callback");
        }

    }
    releaseAssert(m_rank == 0,m_logPretext + "NonNode0 has reached treeDepth");
    return currentData;
}

void MPIRelay::runSlaveLoop()
{
#ifdef USEMPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        return; // master does not participate here

    m_isRunning = true;
    MPI_Status status;

    char* dataBuffer = nullptr;
    size_t bufferSize = 0;
    size_t bufferAlignment = 1;
    while (m_isRunning)
    {
        payloadHeader payload;
        MPI_Recv(&payload, sizeof(payload), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (payload.command == MPICommand::Shutdown)
            break;
        if (traceMessages) logger().log(m_logPretext + "Received command", static_cast<int>(payload.command));
        if (traceMessages) logger().log(m_logPretext + "PayloadSize", static_cast<int>(payload.payloadSize));
        if (traceMessages) logger().log(m_logPretext + "Payload alignment", static_cast<int>(payload.alignment));
        if (payload.payloadSize != 0)
        {
            if (bufferSize < static_cast<size_t>(payload.payloadSize) || bufferAlignment < payload.alignment)
            {
                if (dataBuffer)
                    operator delete[] (dataBuffer,std::align_val_t(bufferAlignment));
                dataBuffer = new (std::align_val_t(payload.alignment)) char[payload.payloadSize];
                bufferAlignment = payload.alignment;
            }

            int64_t bytesReceived = 0;
            while (bytesReceived < payload.payloadSize)
            {
                int toRead;
                MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status,MPI_CHAR,&toRead);
                if (traceMessages)  logger().log(m_logPretext + "reading ", toRead);
                releaseAssert(toRead + bytesReceived <= payload.payloadSize,m_logPretext + "toRead + bytesReceived <= payload.payloadSize");

                MPI_Recv(dataBuffer + bytesReceived, toRead, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bytesReceived += toRead;
            }
            releaseAssert(bytesReceived == payload.payloadSize,m_logPretext + "bytesReceived == payload.payloadSize");
            if (traceMessages) logger().log(m_logPretext + "Done Read ");
        }

        auto it = m_registeredMPICommands.find(payload.command);
        if (it == m_registeredMPICommands.end())
        {
            fprintf(stderr,"[Worker %i] Unknown command %i\n", rank,static_cast<int>(payload.command));
            continue;
        }

        // Execute remote function
        serialDataContainer result = it->second(dataBuffer, payload.payloadSize);
        if (traceMessages) logger().log(m_logPretext + "Done Work ");
        // Send response back
        payload.command = MPICommand::Reply;
        payload.payloadSize = result.ptr ? result.size : 0;
        payload.alignment = result.alignment;
        if (traceMessages) logger().log(m_logPretext + "Sending ",payload.payloadSize);
        MPI_Send(&payload, sizeof(payloadHeader), MPI_CHAR, 0, 0, MPI_COMM_WORLD);

        int64_t sentBytes = 0;
        while (sentBytes < payload.payloadSize)
        {
            int toSend = std::min(payload.payloadSize - sentBytes,(int64_t)maxPayloadSize);
            if (traceMessages) logger().log(m_logPretext + "Send chunk ",toSend);
            MPI_Send(result.ptr.get()+sentBytes, toSend, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            sentBytes += toSend;
        }
        if (traceMessages) logger().log(m_logPretext + "Done send");
    }
    logger().log(m_logPretext + "SlaveLoop quiting",m_rank);
    m_isRunning = false;
    if (dataBuffer)
        operator delete[] (dataBuffer,std::align_val_t(bufferAlignment));
#else
    releaseAssert(false,"runSlaveLoop called without MPI build");
#endif
}



