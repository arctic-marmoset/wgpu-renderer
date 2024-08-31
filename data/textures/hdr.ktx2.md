# hdr.ktx2

For some reason, having mipmaps in textures larger than some size (64x64?)
causes loading to fail with KTX_FILE_DATA_ERROR because of an unexpected mip
level size (see [texture2.c:2292](../../extern/ktx/KTX-Software/lib/texture2.c)).

So, this file has no mipmaps.
