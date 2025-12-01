#ifndef SERIALISE_H
#define SERIALISE_H

//#include "logger.h" cyclic include
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <type_traits>

struct serialDataContainer
{
    std::shared_ptr<char[]> ptr = nullptr;
    size_t size = -1;
    size_t alignment = alignof(char);
};

static inline bool is_aligned(const void* pointer, size_t byte_count)
{ return (uintptr_t)pointer % byte_count == 0; } // not strictly portable. will likely work for all foreseeable architectures

//Number of bytes needed to pad until offset becomes alignment aligned again. Assumes offset=0 is aligned correctly
static constexpr size_t computePaddingBytes(size_t offset, size_t alignment)
{
    return (alignment - offset) % alignment;
}

static inline size_t getAlignment(const void* ptr)
{
    static_assert(std::is_same_v<uintptr_t,unsigned long>);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers" // The builtin is not const but accepts const
    return (1ul << __builtin_ctzl(reinterpret_cast<const uintptr_t>(ptr)));
#pragma GCC diagnostic pop

}



template <typename dataType>
class serialisableArray
{
    size_t m_size;
    const dataType* m_data; // Does not claim ownership
    static constexpr size_t alignment = std::max(alignof(dataType),alignof(size_t));
    static constexpr size_t paddingBytes = computePaddingBytes(sizeof(m_size),alignment);

public:
    serialisableArray(size_t size,const dataType* data){m_size = size; m_data = data;}

    static constexpr size_t getAlignment(){return alignment;}
    size_t getSerialiedSize()
    {
        return m_size*sizeof(dataType) + sizeof(m_size) + paddingBytes;
    }

    serialDataContainer serialise()
    {
        std::shared_ptr<char[]> ptr(new (std::align_val_t(alignment)) char[getSerialiedSize()]);

        std::memcpy(ptr.get(), &m_size, sizeof(m_size));
        std::memcpy(ptr.get()+sizeof(m_size)+paddingBytes, m_data, m_size*sizeof(dataType));

        return {.ptr = ptr,.size = getSerialiedSize(), .alignment = alignment};
    }

    void serialise(char* ptr, size_t  __attribute__ ((unused)) expectedSize)
    {
        assert(expectedSize == getSerialiedSize());
        if (!is_aligned(ptr,alignment))
        {
            fprintf(stderr,"Serialise, alignment is: %zu, expected: %zu\n", ::getAlignment(ptr),alignment);
            __builtin_trap();
        }

        std::memcpy(ptr,&m_size,sizeof(m_size));
        std::memcpy(ptr+sizeof(m_size) + paddingBytes,m_data,m_size*sizeof(dataType));
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
        return size*sizeof(dataType) + sizeof(size) + paddingBytes;
    }

    static void deserialise(char* ptr, const dataType** data)// Stores the pointer where the vector starts in data. Does not involve a copy but ptr must live while data is being used
    {
        char* dataPtr = ptr+sizeof(size_t);
        static_assert(sizeof(*data) == sizeof(dataPtr));
        if (!is_aligned(ptr,alignment))
        {
            fprintf(stderr,"deserialise, alignment is: %zu, expected: %zu\n", ::getAlignment(ptr),alignment);
            __builtin_trap();
        }

        std::memcpy(data,&dataPtr,sizeof(*data)); // in c++ double* A = B is not convertible via a static cast. This is the workaround to reinterpret a char* as a dataType*
    }
    static void deserialise(char* ptr, dataType** data)// Stores the pointer where the vector starts in data. Does not involve a copy but ptr must live while data is being used
    {

        char* dataPtr = ptr+sizeof(size_t);
        static_assert(sizeof(*data) == sizeof(dataPtr));
        if (!is_aligned(ptr,alignment))
        {
            fprintf(stderr,"deserialise, alignment is: %zu, expected: %zu\n", ::getAlignment(ptr),alignment);
            __builtin_trap();
        }

        std::memcpy(data,&dataPtr,sizeof(*data)); // in c++ double* A = B is not convertible via a static cast. This is the workaround to reinterpret a char* as a dataType*
    }

    dataType* getData(){return m_data;}
    size_t getSize(){return m_size;}
};


//No alignment requirements. constructs via copy
template <typename dataType>
serialDataContainer serialiseStruct(const dataType &data)
{
    std::shared_ptr<char[]> ptr(new char[sizeof(data)]);
    std::memcpy(ptr.get(),&data,sizeof(data));
    return {.ptr = ptr,.size = sizeof(data),.alignment = alignof(char)};
}

template <typename dataType>
size_t deserialiseStruct(char* ptr,  dataType &data) // non default constructible objects?, Returns number of bytes consumed
{
    std::memcpy(&data,ptr,sizeof(data));
    return sizeof(data);
}

#endif // SERIALISE_H
