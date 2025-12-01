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

    m_totalNodes = size - 1;
    m_freeNodes = m_totalNodes;

    m_amMaster = m_rank == 0;
    if (m_amMaster)
    {
        registerCommands();
        logger().log("Master node initialized with workers", m_totalNodes);

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
            logger().log("Sending Shutdown", i);
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
    payload.payloadSize = data.size;
    // Determine what it is aligned to here and replicate on the other side.
    // Ideally we actually want to be told this as a memory can accidentally have higher alignment than needed.
    // This leads to uneccessary reallocations on the other side
    payload.alignment = data.alignment;


    if (traceMessages) logger().log("Sending command", static_cast<int>(payload.command));
    if (traceMessages) logger().log("Sending payload.payloadSize", payload.payloadSize);
    if (traceMessages) logger().log("Sending payload.alignment", payload.alignment);

    MPI_Send(&payload, sizeof(payloadHeader), MPI_CHAR, targetNode, 0, MPI_COMM_WORLD);
    int64_t sentBytes = 0;
    while (sentBytes < payload.payloadSize)
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
        logger().log("Slave called waitForAll");
        return;
    }

    MPI_Status status;
    char* dataBuffer = nullptr;
    size_t bufferSize = 0;
    size_t bufferAlignment = 1;

    if (traceMessages) logger().log("Enter Waiting for Nodes", m_totalNodes - m_freeNodes);
    while (m_freeNodes != m_totalNodes)
    {
        if (traceMessages) logger().log("Waiting for Nodes", m_totalNodes - m_freeNodes);
        // Wait for any worker to return
        MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int node = status.MPI_SOURCE;
        if (traceMessages) logger().log("Data from node", node);
        // Determine payload size
        payloadHeader payload;
        MPI_Recv(&payload, sizeof(payload), MPI_CHAR, node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (traceMessages) logger().log("Command", static_cast<int>(payload.command));
        if (traceMessages) logger().log("payloadSize", payload.payloadSize);

        if (bufferSize < static_cast<size_t>(payload.payloadSize) || bufferAlignment < payload.alignment)
        {
            if (dataBuffer)
                delete[] dataBuffer;
            dataBuffer = new (std::align_val_t(payload.alignment)) char[payload.payloadSize];
        }

        // Receive payload
        int64_t bytesReceived = 0;
        while (bytesReceived < payload.payloadSize)
        {
            int toRead;
            MPI_Probe(node, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status,MPI_CHAR,&toRead);
            if (traceMessages) logger().log("Reading", toRead);
            releaseAssert(toRead + bytesReceived <= payload.payloadSize,"toRead + bytesReceived <= payload.payloadSize");

            MPI_Recv(dataBuffer+bytesReceived, toRead, MPI_CHAR, node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            bytesReceived += toRead;
        }
        releaseAssert(bytesReceived == payload.payloadSize,"bytesReceived == payload.payloadSize");
        if (traceMessages) logger().log("Done Read");

        // --- Pop callback for this node ---
        std::function<void(char*)> cb;
        cb = std::move(m_nodeCallBackMap[node]);
        m_nodeCallBackMap.erase(node);
        ++m_freeNodes;

        // Execute returned callback
        if (cb)
            cb(dataBuffer);
        if (traceMessages) logger().log("Done CallBack");
    }
    if (dataBuffer)
        delete[] dataBuffer;
#else
    releaseAssert(false,"waitForAll called without MPI build");
#endif
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
        if (traceMessages) logger().log("Received command", static_cast<int>(payload.command));
        if (traceMessages) logger().log("PayloadSize", static_cast<int>(payload.payloadSize));
        if (traceMessages) logger().log("Payload alignment", static_cast<int>(payload.alignment));

        if (bufferSize < static_cast<size_t>(payload.payloadSize) || bufferAlignment < payload.alignment)
        {
            if (dataBuffer)
                delete[] dataBuffer;
            dataBuffer = new (std::align_val_t(payload.alignment)) char[payload.payloadSize];
        }

        int64_t bytesReceived = 0;
        while (bytesReceived < payload.payloadSize)
        {
            int toRead;
            MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status,MPI_CHAR,&toRead);
            if (traceMessages)  logger().log("reading ", toRead);
            releaseAssert(toRead + bytesReceived <= payload.payloadSize,"toRead + bytesReceived <= payload.payloadSize");

            MPI_Recv(dataBuffer + bytesReceived, toRead, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            bytesReceived += toRead;
        }
        releaseAssert(bytesReceived == payload.payloadSize,"bytesReceived == payload.payloadSize");
        if (traceMessages) logger().log("Done Read ");
        auto it = m_registeredMPICommands.find(payload.command);
        if (it == m_registeredMPICommands.end())
        {
            fprintf(stderr,"[Worker %i] Unknown command %i\n", rank,static_cast<int>(payload.command));
            continue;
        }

        // Execute remote function
        serialDataContainer result = it->second(dataBuffer, payload.payloadSize);
        if (traceMessages) logger().log("Done Work ");
        // Send response back
        payload.command = MPICommand::Reply;
        payload.payloadSize = result.size;
        payload.alignment = result.alignment;
        if (traceMessages) logger().log("Sending ",payload.payloadSize);
        MPI_Send(&payload, sizeof(payloadHeader), MPI_CHAR, 0, 0, MPI_COMM_WORLD);

        int64_t sentBytes = 0;
        while (sentBytes < payload.payloadSize)
        {
            int toSend = std::min(payload.payloadSize - sentBytes,(int64_t)maxPayloadSize);
            if (traceMessages) logger().log("Send chunk ",toSend);
            MPI_Send(result.ptr.get()+sentBytes, toSend, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            sentBytes += toSend;
        }
        if (traceMessages) logger().log("Done send");
    }
    logger().log("SlaveLoop quiting",m_rank);
    m_isRunning = false;
    if (dataBuffer)
        delete[] dataBuffer;
#else
    releaseAssert(false,"runSlaveLoop called without MPI build");
#endif
}



