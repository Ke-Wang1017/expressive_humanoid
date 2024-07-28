
all:
	@(cd sipgen; $(MAKE))
	@(cd siplib; $(MAKE))

install:
	@(cd sipgen; $(MAKE) install)
	@(cd siplib; $(MAKE) install)
	@test -d $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64 || mkdir -p $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64
	cp -f sipconfig.py $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64/sipconfig.py
	cp -f /home/ke/Documents/expressive-humanoid/sip-4.19.3/sipdistutils.py $(DESTDIR)/home/ke/Documents/expressive-humanoid/build/Python38_x64/sipdistutils.py

clean:
	@(cd sipgen; $(MAKE) clean)
	@(cd siplib; $(MAKE) clean)
