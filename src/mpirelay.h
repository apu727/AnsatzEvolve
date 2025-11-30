#ifndef MPIRELAY_H
#define MPIRELAY_H

#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <map>

//TODO put this somewhere better
template <typename dataType>
class serialisableArray
{
    size_t m_size;
    const dataType* m_data; // Does not claim ownership
public:
    serialisableArray(size_t size,const dataType* data){m_size = size; m_data = data;}

    size_t getSerialiedSize()
    {
        return m_size*sizeof(dataType) + sizeof(m_size);
    }

    std::pair<std::shared_ptr<char[]>,size_t> serialise()
    {
        std::shared_ptr<char[]> ptr(new char[getSerialiedSize()]);

        std::memcpy(ptr.get(),&m_size,sizeof(m_size));
        std::memcpy(ptr.get()+sizeof(m_size),m_data,m_size*sizeof(dataType));

        return {ptr,m_size*sizeof(dataType)+sizeof(m_size)};
    }

    void serialise(char* ptr, size_t expectedSize)
    {
        assert(expectedSize == getSerialiedSize());

        std::memcpy(ptr,&m_size,sizeof(m_size));
        std::memcpy(ptr+sizeof(m_size),m_data,m_size*sizeof(dataType));
    }

    static size_t deserialiseSize(char* ptr)// tells you how big the resultant dataType array will be
    {
        size_t size;
        std::memcpy(&size,ptr,sizeof(size));
        return size;
    }
    static size_t deserialise(char* ptr, dataType* data)// Store the data in data. Involves a copy so dont do this if you dont want to. returns the amount of bytes consumed
    {
        size_t size;
        std::memcpy(&size,ptr,sizeof(size));
        std::memcpy(data,ptr+sizeof(size),size*sizeof(dataType));
        return size*sizeof(dataType) + sizeof(size);
    }

    static void deserialise(char* ptr, const dataType** data)// Stores the pointer where the vector starts in data. Does not involve a copy but ptr must live while data is being used
    {
        char* dataPtr = ptr+sizeof(size_t);
        static_assert(sizeof(*data) == sizeof(dataPtr));
        std::memcpy(data,&dataPtr,sizeof(*data)); // in c++ double* A = B is not convertible via a static cast. This is the workaround to reinterpret a char* as a dataType*
    }
    static void deserialise(char* ptr, dataType** data)// Stores the pointer where the vector starts in data. Does not involve a copy but ptr must live while data is being used
    {

        char* dataPtr = ptr+sizeof(size_t);
        static_assert(sizeof(*data) == sizeof(dataPtr));
        std::memcpy(data,&dataPtr,sizeof(*data)); // in c++ double* A = B is not convertible via a static cast. This is the workaround to reinterpret a char* as a dataType*
    }

    dataType* getData(){return m_data;}
    size_t getSize(){return m_size;}
};

template <typename dataType>
std::pair<std::shared_ptr<char[]>,size_t> serialiseStruct(const dataType &data)
{
    std::shared_ptr<char[]> ptr(new char[sizeof(data)]);
    std::memcpy(ptr.get(),&data,sizeof(data));
    return {ptr,sizeof(data)};
}

template <typename dataType>
size_t deserialiseStruct(char* ptr,  dataType &data) // non default constructible objects?, Returns number of bytes consumed
{
    std::memcpy(&data,ptr,sizeof(data));
    return sizeof(data);
}

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
    std::map<MPICommand,std::pair<std::shared_ptr<char[]>, size_t>(*)(char*,size_t)> m_registeredMPICommands; // association between methods to call on the remote and function pointers.

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
    bool IssueCommandToFreeNode(MPICommand comm, char* data, size_t dataSize, std::function<void(char*)> callBack);
    void waitForAll(); // waits for all nodes to be free again
    int getFreeNodeCount(){return m_freeNodes;}; // an estimate. If multiple people are queuing then there may be race conditions. If this becomes a problem, implement a semaphore type behaviour. Does not include the current node

    bool isMaster(){return m_amMaster;};
    void runSlaveLoop();


};


#endif // MPIRELAY_H
