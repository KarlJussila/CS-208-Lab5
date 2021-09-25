/*
 * Simple allocator based on implicit free lists, first fit search,
 * and boundary tag coalescing.
 *
 * Each block has header and footer of the form:
 *
 *      64                  4  3  2  1  0
 *      -----------------------------------
 *     | s  s  s  s  ... s  s  0  0  0  a/f
 *      -----------------------------------
 *
 * where s are the meaningful size bits and a/f is 1
 * if and only if the block is allocated. The list has the following form:
 *
 * begin                                                             end
 * heap                                                             heap
 * -----------------------------------------------------------------------------------------------------------------
 *|  pad | hdr(16:a)|ftr(16:a) | hdr(16:f) | nxt(16:f) | prev(16:f) | zero or more usr blks | ftr(16:f) | hdr(0:a) |
 * -----------------------------------------------------------------------------------------------------------------
 *       |       prologue      | usr block | next      |  previous  |                       | usr block | epilogue |
 *       |        block        | header    | pointer   |  pointer   |                       | footer    | block    |
 *
 * The allocated prologue and epilogue blocks are overhead that
 * eliminate edge conditions during coalescing.
 *
 * The free and allocated blocks sit in arbitrary ordering in the heap. Every free block has a next pointer and a previous pointer,
 * stored in the first and second eight byte words of the payload, respectively. The free list has an arbitrary order. The last free
 * block's next points to NULL, and the first free block's previous points to NULL. rootp, a global variable, holds a pointer to the
 * first block in the free list. To add a new block to the free list, the root is pointed at the new block, and the new block's next
 * is pointed at what the root was pointing at before. To remove a free block from the list, the block's next is pointed at its
 * previous, and its previous is pointed at its next. If the block is the first in the free list, rootp is pointed at the block's next.
 * Free blocks are removed from the list when they are allocated, and allocated blocks are added to the list when they are freed.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

#include "mm.h"
#include "memlib.h"

/*********************************************************
 * NOTE: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "BOBBUI2",
    /* First member's full name */
    "Karl Jussila",
    /* First member's email address */
    "jussilak@carleton.edu",
    /* Second member's full name (leave blank if none) */
    "Justin Yamada",
    /* Second member's email address (leave blank if none) */
    "yamadaj@carleton.edu"
};

/* Basic constants and macros */
#define WSIZE       8       /* word size (bytes) */
#define DSIZE       16      /* doubleword size (bytes) */
#define CHUNKSIZE  (1<<12)  /* initial heap size (bytes) */
#define OVERHEAD    16      /* overhead of header and footer (bytes) */
#define MINBLKSIZE  32

/* NOTE: feel free to replace these macros with helper functions and/or
 * add new ones that will be useful for you. Just make sure you think
 * carefully about why these work the way they do
 */

/* Pack a size and allocated bit into a word */
#define PACK(size, alloc)  ((size) | (alloc))

/* Read and write a word at address p */
#define GET(p)       (*(size_t *)(p))
#define PUT(p, val)  (*(size_t *)(p) = (val))

/* Perform unscaled pointer arithmetic */
#define PADD(p, val) ((char *)(p) + (val))
#define PSUB(p, val) ((char *)(p) - (val))

/* Read the size and allocated fields from address p */
#define GET_SIZE(p)  (GET(p) & ~0xf)
#define GET_ALLOC(p) (GET(p) & 0x1)

/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp)       (PSUB(bp, WSIZE))
#define FTRP(bp)       (PADD(bp, GET_SIZE(HDRP(bp)) - DSIZE))

/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp)  (PADD(bp, GET_SIZE(HDRP(bp))))
#define PREV_BLKP(bp)  (PSUB(bp, GET_SIZE((PSUB(bp, DSIZE)))))

/* Set the header size in bp */
#define SET_HDR(bp, size, a) (PUT(HDRP(bp), PACK(size, a)))

/* Set footer size for bp */
#define SET_FTR(bp, size, a) (PUT(FTRP(bp), PACK(size, a)))

/* Set previous free block pointer */
#define SET_PREV_P(bp, pp) (PUT_P(PADD(bp, WSIZE), pp))

/* Set next free block pointer */
#define SET_NEXT_P(bp, np) (PUT_P(bp, np))

/* Read and write a pointer at address p */
#define GET_P(p)       ((void**) (p))
#define PUT_P(p, val)  (*(void**)(p) = (val))

/* Given block ptr bp, compute address of the previous and next block ptr */
#define GET_PREV_P(bp) (*(void**) GET_P(PADD(bp, WSIZE)))
#define GET_NEXT_P(bp) (*(void**) GET_P(PADD(bp,0)))


/* Global variables */

// Pointer to first block
static void *heap_start = NULL;
// Pointer to first free block
static void *rootp = NULL;

/* Function prototypes for internal helper routines */

static bool check_heap(int lineno);
static void print_heap();
static void print_block(void *bp);
static bool check_block(int lineno, void *bp);
static void *extend_heap(size_t size);
static void *find_fit(size_t asize);
static void *coalesce(void *bp);
static void place(void *bp, size_t asize);
static size_t max(size_t x, size_t y);
// added functions
static void remove_free_block(void* bp);
static void add_free_block(void* bp);
static void print_free_list();

/*
 * mm_init creates an empty heap with a prologue header and footer, and epilogue header
 * expands the heap to include a free block after creating its basic structure
 */
int mm_init(void) {

    // initializing rootp to NULL avoids errors
    rootp = NULL;

    /* create the initial empty heap */
    if ((heap_start = mem_sbrk(4 * WSIZE)) == NULL) {
        return -1;
    }

    PUT(heap_start, 0);            /* root pointer */
    PUT(PADD(heap_start, WSIZE), PACK(OVERHEAD, 1));  /* prologue header */
    PUT(PADD(heap_start, DSIZE), PACK(OVERHEAD, 1));  /* prologue footer */
    PUT(PADD(heap_start, WSIZE + DSIZE), PACK(0, 1));   /* epilogue header */

    heap_start = PADD(heap_start, DSIZE); /* start the heap at the (size 0) payload of the prologue block */

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE / WSIZE) == NULL){
        return -1;
    }

    return 0;
}

/* Removes free block pointed to by bp from the free list */
void remove_free_block(void* bp) {

    size_t size = GET_SIZE(HDRP(bp));

    // set the header and footer of the new block
    SET_HDR(bp, size, 1);
    SET_FTR(bp, size, 1);

    void *next_free_bp = GET_NEXT_P(bp);
    void *prev_free_bp = GET_PREV_P(bp);

    // set the pointers of bp's prev and next to each other if they exist
    if (prev_free_bp) {
        SET_NEXT_P(prev_free_bp, next_free_bp);
    } else {
        rootp = next_free_bp;
    }
    if (next_free_bp) {
        SET_PREV_P(next_free_bp, prev_free_bp);
    }

}

/* adds the free block pointed to by bp to the free list */
void add_free_block(void* bp) {

    size_t size = GET_SIZE(HDRP(bp));

    // set the header and footer of the new free block
    SET_HDR(bp, size, 0);
    SET_FTR(bp, size, 0);

    // add the free block to the free list
    SET_NEXT_P(bp, rootp);
    SET_PREV_P(bp, NULL);

    // if the free block is not the first, run this command
    if (rootp) {
        SET_PREV_P(rootp, bp);
    }

    // let the root point to the new free block
    rootp = bp;

}

/* Test function to help us navigate through individual block errors  */
void print_free_list() {
    printf("----------------------\n");
    void *cur_blk = rootp;
    while (cur_blk) {
        print_block(cur_blk);
        printf("Next: %p\n", GET_NEXT_P(cur_blk));
        printf("Prev: %p\n", GET_PREV_P(cur_blk));
        cur_blk = GET_NEXT_P(cur_blk);
    }
    printf("----------------------\n");
}

/*
 * mm_malloc -- this function will allocate "size" space for data to be placed within the heap
 */
void *mm_malloc(size_t size) {
    size_t asize;      /* adjusted block size */
    size_t extendsize; /* amount to extend heap if no fit */
    char *bp;

    /* Ignore spurious requests */
    if (size <= 0)
        return NULL;

    /* Adjust block size to include overhead and alignment reqs. */
    if (size <= DSIZE) {
        asize = DSIZE + OVERHEAD;
    } else {
        /* Add overhead and then round up to nearest multiple of double-word alignment */
        asize = DSIZE * ((size + (OVERHEAD) + (DSIZE - 1)) / DSIZE);
    }

    /* Search the free list for a fit */
    if ((bp = find_fit(asize)) != NULL) {
        place(bp, asize);
        return bp;
    }

    /* No fit found. Get more memory and place the block */
    extendsize = max(asize, CHUNKSIZE);

    if ((bp = extend_heap(extendsize / WSIZE)) == NULL)
        return NULL;

    place(bp, asize);
    return bp;
}

/*
 * mm_free -- if bp is not null and it is not already allocated then free the blocks and coalesece with
 * any adjacent free blocks
 */
void mm_free(void *bp) {

    // two if statements to avoid mm_free being called with bad requests
    if (bp == NULL) {
        return;
    }
    if (!GET_ALLOC(HDRP(bp))) {
        return;
    }

    // add a free block to the free list
    add_free_block(bp);

    // coalesce the newly created free block with adjacent blocks
    coalesce(bp);
    return;

}

/*
 * EXTRA CREDIT
 * mm_realloc realloc takes a pointer to a block and a size and returns a pointer to a block of the requested size
 * containing the information that was in the provided block
*/
void *mm_realloc(void *ptr, size_t size) {
    // TODO: implement this function for EXTRA CREDIT

    // If the pointer is null, just malloc a new block with payload of size
    if (ptr == NULL) {
        return mm_malloc(size);
    }

    // If the size requested is 0, just free the block
    if (size == 0) {
        mm_free(ptr);
        return NULL;
    }

    // Get total size of the requested block, including overhead
    size_t asize = size + DSIZE;

    // Save current size of the block
    size_t cur_size = GET_SIZE(HDRP(ptr));

    // If the requested size is the same or smaller than the current size, return the current block as it is
    if (asize <= cur_size) {
        return ptr;
    }

    size_t next_blk_size = GET_SIZE(HDRP(NEXT_BLKP(ptr)));
    size_t total_space = cur_size + next_blk_size;

    // If the next block is allocated and the combined space of the current and next blocks can fit asize
    if(!GET_ALLOC(HDRP(NEXT_BLKP(ptr))) && total_space >= asize) {
        // Remove the next block from the freelist
        remove_free_block(NEXT_BLKP(ptr));

        // Set the header and footer to merge the current and next block
        SET_HDR(ptr, total_space, 1);
        SET_FTR(ptr, total_space, 1);

        // Return pointer to the current block
        return ptr;
    }

    // If all else fails, malloc a new block for the requested space
    void *new_ptr = mm_malloc(asize);
    memcpy(new_ptr, ptr, cur_size);
    mm_free(ptr);
    return new_ptr;
}


/* The remaining routines are internal helper routines */


/*
 * place -- Place block of asize bytes at start of free block bp
 *          and split the block by setting the correct length header
 *          and footer and adding the split block into the free list
 *          in addition to coalsecing the split free block
 *
 * Takes a pointer to a free block and the size of block to place inside it
 * Returns nothing
 *
 */
static void place(void *bp, size_t asize) {

    size_t prev_size = GET_SIZE(HDRP(bp));

    // check if there is enough additional space to split another block
    if (prev_size >= asize + MINBLKSIZE) {

        // Set previous and next pointer of the split block to the previous and next pointer of the block being allocated
        SET_HDR(bp, asize, 1);
        SET_FTR(bp, asize, 1);

        // removes the block from the free list in preparation for data being inputted
        remove_free_block(bp);

        void *split_blk = NEXT_BLKP(bp);

        // set the header and footer of the new split free block
        SET_HDR(split_blk, prev_size - asize, 0);
        SET_FTR(split_blk, prev_size - asize, 0);

        // create free space and check if adjacent free blocks need to be coalesced
        add_free_block(split_blk);
        coalesce(split_blk);
    } else {
        // remove free block so that data can be placed
        remove_free_block(bp);
    }

}

/*
 * coalesce -- Boundary tag coalescing.
 * Takes a pointer to a free block
 * Return ptr to coalesced block
 * If bp is already allocated just return bp
 */
static void *coalesce(void *bp) {

    // if bp is already allocated return bp
    if (GET_ALLOC(HDRP(bp))) {
        return bp;
    }


    // blocks adjacent to bp
    void *prev_bp = PREV_BLKP(bp);
    void *next_bp = NEXT_BLKP(bp);

    // allocation status of blocks adjacent to bp
    size_t left_alloc = GET_ALLOC(HDRP(PREV_BLKP(bp)));
    size_t right_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));

    // size of bp
    size_t bp_size = GET_SIZE(HDRP(bp));

    // "if" statement that checks the left adjacent block if allocated
    if (!left_alloc) {

        // add size to accommodate
        bp_size += GET_SIZE(HDRP(prev_bp));

        // remove the free block bp and the free block prev_bp in order
        // to reset headers and footers and be added to free list
        remove_free_block(bp);
        remove_free_block(prev_bp);
        SET_HDR(prev_bp, bp_size, 0);
        SET_FTR(bp, bp_size, 0);

        // "if" statement that sees if the right adjacent block is free after confirming the left adjacent block is free
        if (!right_alloc){

            // similar to the left side remove free block these four lines will combine the prev, bp, and next
            // and set the appropriate hdr and ftr
            bp_size += GET_SIZE(HDRP(next_bp));
            remove_free_block(next_bp);
            SET_HDR(prev_bp, bp_size, 0);
            SET_FTR(next_bp, bp_size, 0);
        }


        // with the correct header and footer set, add the new free block into the free list
        add_free_block(prev_bp);
        return prev_bp;

    // situation in which only the right side is free, set the size from bp to the block right
    // adjacent to bp. Additionally set the appropriate header and footer after adding the free
    // block
    } else if(!right_alloc) {
        bp_size += GET_SIZE(HDRP(next_bp));
        remove_free_block(bp);
        remove_free_block(next_bp);
        SET_HDR(bp, bp_size, 0);
        SET_FTR(next_bp, bp_size, 0);
        add_free_block(bp);
    }

    // return the pointer to the newly coalesced block
    return bp;
}


/*
 * find_fit - Find a fit for a block with asize bytes
 */
static void *find_fit(size_t asize) {
    /* search from the start of the free list to the end */

    void *cur_block = rootp;

    // while the current block is not null, find a block that is bigger or equal to the requested size
    // if found return the block but if not return null
    while(cur_block != NULL) {
        if (asize <= GET_SIZE(HDRP(cur_block))) {
            return cur_block;
        }

        cur_block = GET_NEXT_P(cur_block);
    }

    return NULL;  /* no fit found */
}

/*
 * extend_heap - Extend heap with free block and return its block pointer
 */
static void *extend_heap(size_t words) {
    char *bp;
    size_t size;

    /* Allocate an even number of words to maintain alignment */
    size = words * WSIZE;
    if (words % 2 == 1)
        size += WSIZE;

    // printf("extending heap to %zu bytes\n", mem_heapsize());
    if ((long)(bp = mem_sbrk(size)) < 0)
        return NULL;

    /* Initialize free block header/footer and the epilogue header */
    PUT(HDRP(bp), PACK(size, 0));         /* free block header */
    PUT(FTRP(bp), PACK(size, 0));         /* free block footer */

    // add bp to the free block list
    add_free_block(bp);

    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1)); /* new epilogue header */

    /* Coalesce if the previous block was free */
    return coalesce(bp);
}

/*
 * check_heap -- Performs basic heap consistency checks for an implicit free list allocator
 * and prints out all blocks in the heap in memory order.
 * Checks include proper prologue and epilogue, alignment, and matching header and footer
 * Takes a line number (to give the output an identifying tag).
 */
static bool check_heap(int line) {
    char *bp;

    if ((GET_SIZE(HDRP(heap_start)) != DSIZE) || !GET_ALLOC(HDRP(heap_start))) {
        printf("(check_heap at line %d) Error: bad prologue header\n", line);
        return false;
    }

    for (bp = heap_start; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
        if (!check_block(line, bp)) {
            return false;
        }
    }

    if ((GET_SIZE(HDRP(bp)) != 0) || !(GET_ALLOC(HDRP(bp)))) {
        printf("(check_heap at line %d) Error: bad epilogue header\n", line);
        return false;
    }

    return true;
}

/*
 * check_block -- Checks a block for alignment and matching header and footer
 */
static bool check_block(int line, void *bp) {
    if ((size_t)bp % DSIZE) {
        printf("(check_heap at line %d) Error: %p is not double-word aligned\n", line, bp);
        return false;
    }
    if (GET(HDRP(bp)) != GET(FTRP(bp))) {
        printf("(check_heap at line %d) Error: header does not match footer\n", line);
        return false;
    }
    return true;
}

/*
 * print_heap -- Prints out the current state of the implicit free list
 */
static void print_heap() {
    char *bp;

    printf("Heap (%p):\n", heap_start);

    for (bp = heap_start; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
        print_block(bp);
    }

    print_block(bp);
}

/*
 * print_block -- Prints out the current state of a block
 */
static void print_block(void *bp) {
    size_t hsize, halloc, fsize, falloc;

    hsize = GET_SIZE(HDRP(bp));
    halloc = GET_ALLOC(HDRP(bp));
    fsize = GET_SIZE(FTRP(bp));
    falloc = GET_ALLOC(FTRP(bp));

    if (hsize == 0) {
        printf("%p: End of free list\n", bp);
        return;
    }

    printf("%p: header: [%ld:%c] footer: [%ld:%c]\n", bp,
       hsize, (halloc ? 'a' : 'f'),
       fsize, (falloc ? 'a' : 'f'));
}

/*
 * max: returns x if x > y, and y otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}
