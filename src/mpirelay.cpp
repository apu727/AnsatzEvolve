#include "mpirelay.h"
#include "logger.h"

#include "hamiltonianmatrix.h"

#ifdef USEMPI
#include <mpi.h>
#endif


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

bool MPIRelay::IssueCommandToFreeNode(MPICommand comm, char *data, size_t dataSize, std::function<void (char *)> callBack)
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

    int cmdInt = static_cast<int>(comm);
    MPI_Send(&cmdInt, 1, MPI_INT, targetNode, 0, MPI_COMM_WORLD);
    MPI_Send(data, dataSize, MPI_CHAR, targetNode, 0, MPI_COMM_WORLD);
    logger().log("Sending command", cmdInt);
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
    std::vector<char> dataBuffer;
    logger().log("Enter Waiting for Nodes", m_totalNodes - m_freeNodes);
    while (m_freeNodes != m_totalNodes)
    {
        logger().log("Waiting for Nodes", m_totalNodes - m_freeNodes);
        // Wait for any worker to return
        MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int node = status.MPI_SOURCE;

        // Determine payload size
        MPI_Count byteCount;
        MPI_Get_elements_x(&status, MPI_CHAR, &byteCount);

        if (dataBuffer.size() < static_cast<size_t>(byteCount))
            dataBuffer.resize(byteCount);

        // Receive payload
        MPI_Recv(dataBuffer.data(), byteCount, MPI_CHAR, node, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // --- Pop callback for this node ---
        std::function<void(char*)> cb;
        cb = std::move(m_nodeCallBackMap[node]);
        m_nodeCallBackMap.erase(node);
        ++m_freeNodes;

        // Execute returned callback
        if (cb)
            cb(dataBuffer.data());
    }
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

    std::vector<char> dataBuffer;
    while (m_isRunning)
    {
        int cmd;
        MPI_Recv(&cmd, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPICommand command = static_cast<MPICommand>(cmd);

        if (command == MPICommand::Shutdown)
            break;
        logger().log("Received command", cmd);
        // Probe second message to determine payload size
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Count byteCount;                     //long long
        MPI_Get_elements_x(&status, MPI_CHAR, &byteCount);

        if (dataBuffer.size() < static_cast<size_t>(byteCount))
            dataBuffer.resize(byteCount);

        MPI_Recv(dataBuffer.data(), byteCount, MPI_CHAR, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto it = m_registeredMPICommands.find(command);
        if (it == m_registeredMPICommands.end())
        {
            fprintf(stderr,"[Worker %i] Unknown command %i\n", rank,cmd);
            continue;
        }

        // Execute remote function
        std::pair<std::shared_ptr<char[]>, size_t> result = it->second(&dataBuffer[0], byteCount);

        // Send response back
        MPI_Send(result.first.get(), result.second, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    logger().log("SlaveLoop quiting",m_rank);
    m_isRunning = false;
#else
    releaseAssert(false,"runSlaveLoop called without MPI build");
#endif
}



