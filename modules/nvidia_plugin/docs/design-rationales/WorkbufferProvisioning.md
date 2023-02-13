# Workbuffer provisioning


## Concerns

1. Some operators need to allocate a temporary workbuffer for the purpose of transferring run-time data to the devices.
2. The workbuffer is network specific and can be estimated during operator construction
3. The scope of Execute method is not sufficient for allocating/deallocating work buffers because the kernels are executed asynchronously and may run when the Execute method has already deallocated the buffers.
4. The operator instance cannot allocate the workbuffer in the constructor, since the same instance is used for multiple infer requests, which may run in parallel.
5. A kernel may need both mutable and immutable workbuffers. The immutable work buffers keep their content through the scope of  outer network, while mutable have to be updated with every infer.
6. Frequent allocation/deallocation may fragment device memory

## Considerations

* From the Memory Manager point of view
    - a mutable workbuffer is an input/output tensor with its time box starting on T and lasting till T (1 slot in total)
    - an immutable workbuffer is an input/output tensor a  time box spanning through the nework scope
    - It's a special case which should work (but not tested in MemorySolver) when producerIndex is equal to lastConsumerIndex.
      Trying to use other values will keep buffer alive wirthout a reason.
* Use several workbufers (i.e. a vector) the same way as for inputs and outputs could be as well beneficial for the following reasons:
    - The smaller workbuffer is, the easier to allocate it in a gaps between other memory segments,
      the less the risk to allocate it after all other tensors potentially increasing total size of memory blob.
    - a proper data alignment for free
* `OperationBuffersExtractor` manages allocations of tensor indices, so we need to involve it somehow.
    - After creation of each operation we can request workbuffer sizes and add them toOperationBuffersExtractor
      (passing also operation index and getting workbuffer indices from the same call).
    - Inside of OperationBufferExtractor and in its interface we can treat work-buffers as usual mutable buffers.
    - Any operation in fact doesn't require buffer indices in constructor.
      So, lets add a method OperationBase::setBufferIndices(inputs, outputs, workbuffers).
* We can handle workbuffers the same way as mutable tensors in OperationBuffersExtractor. This way we will not need additional method in MemoryManager.

## Design Decisions
1. A workbuffer request contains vectors of sizes for shared immutable and mutable buffers
2. Immutable workbuffers are initialized on network compilation
3. `IOperationExec` declares method `GetWorkbuffersSize()`
4. `IOperationExec` declares method `InitSharedImmutableWorkbuffers()`
5. `OperatorBase` implements methods `GetWorkbuffersSize()` returning empty request and `InitSharedImmutableWorkbuffers()` doing nothing
6. Signature of `Execute` will be altered to accommodate work buffer pointers
7. `ExecutableNetwork`, on creating operation instances, queries every operation instance for the required work buffer sizes and uses
   `OperationBuffersExtractor` to collects all these requests
7. `ExecutableNetwork`, calls `MemoryModelBuilder` which will also allocate slots for the work buffers
8. `MemoryManager` implements `workBuffers()` method returning vectors of work buffer pointers for the given operator
9. `InferRequest` in the startPipeline method gets work buffer pointers from the memory manager and passes it to the `Execute` method

