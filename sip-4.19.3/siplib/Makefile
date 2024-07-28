TARGET = fbxsip.so
OFILES = siplib.o apiversions.o descriptors.o qtlib.o threads.o objmap.o voidptr.o array.o bool.o
HFILES = sip.h sipint.h array.h 

CC = gcc
CXX = g++
LINK = g++
CPPFLAGS = -DNDEBUG -I. -I/home/ke/miniconda3/envs/humanoid/include/python3.8
CFLAGS = -pipe -fPIC -O2 -Wall -W
CXXFLAGS = -pipe -fPIC -O2 -Wall -W
LFLAGS = -shared -Wl,--version-script=fbxsip.exp
LIBS = 
.SUFFIXES: .c .o .cpp .cc .cxx .C


.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.C.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

all: $(TARGET)

$(OFILES): $(HFILES)

$(TARGET): $(OFILES)
	@echo '{ global: PyInit_fbxsip; local: *; };' > fbxsip.exp
	$(LINK) $(LFLAGS) -o $(TARGET) $(OFILES) $(LIBS)

install: $(TARGET)
	@test -d $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64 || mkdir -p $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64
	cp -f $(TARGET) $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64/$(TARGET)
	strip $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64/$(TARGET)
	@test -d $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64 || mkdir -p $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64
	cp -f /home/ke/Documents/expressive-humanoid/sip-4.19.3/siplib/sip.h $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64/sip.h
	@test -d $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64 || mkdir -p $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64
	cp -f /home/ke/Documents/expressive-humanoid/sip-4.19.3/sip.pyi $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64/sip.pyi

clean:
	-rm -f $(TARGET)
	-rm -f siplib.o
	-rm -f apiversions.o
	-rm -f descriptors.o
	-rm -f qtlib.o
	-rm -f threads.o
	-rm -f objmap.o
	-rm -f voidptr.o
	-rm -f array.o
	-rm -f bool.o
	-rm -f fbxsip.exp
