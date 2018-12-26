program reaxff2lib
!
!  This is a utility code to convert the library files distributed with the ReaxFF code
!  from Adri van Duin to a GULP format library file.
!
!  NB: This utility is provided to try to make life easier, but should not be taken as
!      100% fool proof & so should checks should be made. Note that some parameters that
!      are contained with the control file for the ReaxFF code will need to be set
!      separately.
!
!  For valence potentials, if the first term (vka) is less than zero then the potential is
!  excluded from the output. This is based on a line in the original code that skips this
!  case, but this behaviour may be different in different versions.
!
!  To compile:
!
!  f90 -o reaxff2lib reaxff2lib.f90
!
!  where f90 represents your favourite f90 compatible compiler. g95 is used by me & so I
!  can vouch for it.
!
!  Use:
!
!  ./reaxff2lib < ffield > reaxff_my_system.lib
!
!  Julian Gale, NRI, Curtin University, November 2010
!
  implicit none
!
!  Declare variables
!
  character(len=132)              :: line
  character(len=2), allocatable   :: symbol(:)
  integer*4                       :: i
  integer*4                       :: j
  integer*4                       :: n
  integer*4                       :: n1
  integer*4                       :: n2
  integer*4                       :: n3
  integer*4                       :: n4
  integer*4                       :: nx
  integer*4                       :: nangle
  integer*4                       :: nbond
  integer*4                       :: nhb
  integer*4                       :: nod
  integer*4                       :: npar
  integer*4                       :: nspec
  integer*4                       :: ntors
  integer*4, allocatable          :: nhbspec(:,:)
  integer*4, allocatable          :: nbondspec(:,:)
  integer*4, allocatable          :: nodspec(:,:)
  integer*4, allocatable          :: nanglespec(:,:)
  integer*4, allocatable          :: ntorsspec(:,:)
  logical                         :: first
  real*8                          :: dummy1
  real*8                          :: dummy2
  real*8                          :: dummy3
  real*8                          :: dummy4
  real*8                          :: mass
  real*8                          :: reaxff_par(39)
  real*8,    allocatable          :: reaxff1_morse(:,:)
  real*8,    allocatable          :: reaxff_chi(:)
  real*8,    allocatable          :: reaxff_mu(:)
  real*8,    allocatable          :: reaxff_gamma(:)
  real*8,    allocatable          :: reaxff1_lonepair(:,:)
  real*8,    allocatable          :: reaxff1_radii(:,:)
  real*8,    allocatable          :: reaxff1_under(:)
  real*8,    allocatable          :: reaxff1_over(:,:)
  real*8,    allocatable          :: reaxff1_valence(:,:)
  real*8,    allocatable          :: reaxff1_angle(:,:)
  real*8,    allocatable          :: reaxff2_bond(:,:)
  real*8,    allocatable          :: reaxff2_bo(:,:)
  real*8,    allocatable          :: reaxff2_over(:)
  real*8,    allocatable          :: reaxff2_morse(:,:)
  real*8,    allocatable          :: reaxff3_angle(:,:)
  real*8,    allocatable          :: reaxff3_hbond(:,:)
  real*8,    allocatable          :: reaxff3_penalty(:)
  real*8,    allocatable          :: reaxff3_conj(:)
  real*8,    allocatable          :: reaxff4_torsion(:,:)
!
!  Read ReaxFF file
!
!  Dummy line
  read(5,'(a)') line
!***************
!  Parameters  *
!***************
  read(5,*) npar
  if (npar.gt.39) then
    write(6,'('' Error: Number of parameters greater than 39!'')')
    stop
  endif
  do i = 1,npar
    read(5,*) reaxff_par(i)
  enddo
!************************
!  Species information  *
!************************
  read(5,*) nspec
!  3 x dummy line
  read(5,'(a)') line
  read(5,'(a)') line
  read(5,'(a)') line
!
!  Allocate species arrays
!
  allocate(symbol(0:nspec))
  allocate(reaxff_chi(nspec))
  allocate(reaxff_mu(nspec))
  allocate(reaxff_gamma(nspec))
  allocate(reaxff1_morse(4,nspec))
  allocate(reaxff1_lonepair(2,nspec))
  allocate(reaxff1_radii(3,nspec))
  allocate(reaxff1_valence(4,nspec))
  allocate(reaxff1_under(nspec))
  allocate(reaxff1_over(4,nspec))
  allocate(reaxff1_angle(2,nspec))
!
!  Assign species 0 to be X
!
  nx = 0
  symbol(0) = 'X '
!
!  Loop over species
!
  do n = 1,nspec
!
!  First line
!
    read(5,*) symbol(n),reaxff1_radii(1,n),reaxff1_valence(1,n),mass,reaxff1_morse(3,n),reaxff1_morse(2,n), &
      reaxff_gamma(n),reaxff1_radii(2,n),reaxff1_valence(3,n)
!
!  Compute terms dependent on this line
!
    reaxff1_lonepair(1,n) = 0.5d0*(reaxff1_valence(3,n) - reaxff1_valence(1,n))
!
!  Second line
!
    read(5,*) reaxff1_morse(1,n),reaxff1_morse(4,n),reaxff1_valence(4,n),reaxff1_under(n),dummy1,reaxff_chi(n), &
      reaxff_mu(n),dummy2
!
!  Third line
!
    read(5,*) reaxff1_radii(3,n),reaxff1_lonepair(2,n),dummy1,reaxff1_over(2,n),reaxff1_over(1,n),reaxff1_over(3,n), &
      dummy2,dummy3
!
!  Fourth line
!
    read(5,*) reaxff1_over(4,n),reaxff1_angle(1,n),dummy1,reaxff1_valence(2,n),reaxff1_angle(2,n),dummy2,dummy3,dummy4
!
!  Set pointer to X species if present
!
    if (symbol(n).eq.'X ') nx = n
  enddo
!
!  Read number of bonds
!
  read(5,*) nbond
!  Dummy line
  read(5,'(a)') line
!
!  Allocate arrays
!
  allocate(nbondspec(2,nbond))
  allocate(reaxff2_bond(5,nbond))
  allocate(reaxff2_bo(9,nbond))
  allocate(reaxff2_over(nbond))
!
!  Read bond parameters
!
  do n = 1,nbond
!
!  Read first line
!
    read(5,*) nbondspec(1,n),nbondspec(2,n),reaxff2_bond(1,n),reaxff2_bond(2,n),reaxff2_bond(3,n), &
      reaxff2_bond(4,n),reaxff2_bo(5,n),reaxff2_bo(7,n),reaxff2_bo(6,n),reaxff2_over(n)
!
!  Read second line
!
    read(5,*) reaxff2_bond(5,n),reaxff2_bo(3,n),reaxff2_bo(4,n),dummy1,reaxff2_bo(1,n),reaxff2_bo(2,n), &
      reaxff2_bo(8,n),reaxff2_bo(9,n)
!
!  If reaxff2_bo(3,n) = 1 needs to be set to 0 for GULP since this is a dummy value
!
    if (abs(reaxff2_bo(3,n)-1.0d0).lt.1.0d-12) reaxff2_bo(3,n) = 0.0d0
!
!  If reaxff2_bo(5,n) < 0 needs to be set to 0 for GULP since this is a dummy value
!
!    if (reaxff2_bo(5,n).lt.0.0d0) reaxff2_bo(5,n) = 0.0d0
  enddo
!
!  Read number of off-diagonal terms
!
  read(5,*) nod
!
!  Allocate arrays
!
  allocate(nodspec(2,nod))
  allocate(reaxff2_morse(6,nod))
!
!  Read off-diagonal parameters
!
  do n = 1,nod
    read(5,*) nodspec(1,n),nodspec(2,n),reaxff2_morse(1,n),reaxff2_morse(3,n),reaxff2_morse(2,n), &
      reaxff2_morse(4,n),reaxff2_morse(5,n),reaxff2_morse(6,n)
  enddo
!
!  Read number of angle terms
!
  read(5,*) nangle
!
!  Allocate angle arrays
!
  allocate(nanglespec(3,nangle))
  allocate(reaxff3_angle(5,nangle))
  allocate(reaxff3_penalty(nangle))
  allocate(reaxff3_conj(nangle))
!
!  Read in angle and penalty parameters
!
  do n = 1,nangle
    read(5,*) nanglespec(2,n),nanglespec(1,n),nanglespec(3,n),reaxff3_angle(1,n),reaxff3_angle(2,n), &
      reaxff3_angle(3,n),reaxff3_conj(n),reaxff3_angle(5,n),reaxff3_penalty(n),reaxff3_angle(4,n)
  enddo
!
!  Read number of torsion terms
!
  read(5,*) ntors
!
!  Allocate torsion arrays
!
  allocate(ntorsspec(4,ntors))
  allocate(reaxff4_torsion(5,ntors))
!
!  Read torsion parameters
!
  do n = 1,ntors
    read(5,*) (ntorsspec(i,n),i=1,4),reaxff4_torsion(1,n),reaxff4_torsion(2,n),reaxff4_torsion(3,n), &
      reaxff4_torsion(4,n),reaxff4_torsion(5,n),dummy1,dummy2
  enddo
!
!  Read number of hydrogen bond terms
!
  read(5,*) nhb
!
!  Allocate torsion arrays
!
  allocate(nhbspec(3,nhb))
  allocate(reaxff3_hbond(4,nhb))
!
!  Read hydrogen bond parameters
!
  do n = 1,nhb
    read(5,*) nhbspec(2,n),nhbspec(1,n),nhbspec(3,n),(reaxff3_hbond(i,n),i=1,4)
  enddo
!
!  Output GULP library file
!
  write(6,'(''#'')')
  write(6,'(''#  ReaxFF force field'')')
  write(6,'(''#'')')
  write(6,'(''#  Original paper:'')')
  write(6,'(''#'')')
  write(6,'(''#  A.C.T. van Duin, S. Dasgupta, F. Lorant and W.A. Goddard III,'')')
  write(6,'(''#  J. Phys. Chem. A, 105, 9396-9409 (2001)'')')
  write(6,'(''#'')')
  write(6,'(''#'')')
  write(6,'(''#  Cutoffs for VDW & Coulomb terms'')')
  write(6,'(''#'')')
  write(6,'(''reaxFFvdwcutoff '',f12.4)') reaxff_par(13)
  write(6,'(''reaxFFqcutoff   '',f12.4)') reaxff_par(13)
  write(6,'(''#'')')
  write(6,'(''#  Bond order threshold - check anglemin as this is cutof2 given in control file'')')
  write(6,'(''#'')')
  write(6,'(''reaxFFtol       '',f12.10,'' 0.001'')') 0.01d0*reaxff_par(30)
  write(6,'(''#'')')
  write(6,'(''#  Species independent parameters '')')
  write(6,'(''#'')')
  write(6,'(''reaxff0_bond    '',2(1x,f12.6))') reaxff_par(1),reaxff_par(2)
  write(6,'(''reaxff0_over    '',5(1x,f12.6))') reaxff_par(33),reaxff_par(32),reaxff_par(7),reaxff_par(9),reaxff_par(10)
  write(6,'(''reaxff0_valence '',4(1x,f12.6))') reaxff_par(15),reaxff_par(34),reaxff_par(17),reaxff_par(18)
  write(6,'(''reaxff0_penalty '',3(1x,f12.6))') reaxff_par(20),reaxff_par(21),reaxff_par(22)
  write(6,'(''reaxff0_torsion '',4(1x,f12.6))') reaxff_par(24),reaxff_par(25),reaxff_par(26),reaxff_par(28)
  write(6,'(''reaxff0_vdw     '',1(1x,f12.6))') reaxff_par(29)
  write(6,'(''reaxff0_lonepair'',1(1x,f12.6))') reaxff_par(16)
  write(6,'(''#'')')
  write(6,'(''#  Species parameters '')')
  write(6,'(''#'')')
  write(6,'(''reaxff1_radii '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',3(f8.4,1x))') symbol(n),(reaxff1_radii(i,n),i=1,3)
    endif
  enddo
  write(6,'(''reaxff1_valence '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),(reaxff1_valence(i,n),i=1,4)
    endif
  enddo
  write(6,'(''reaxff1_over '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),(reaxff1_over(i,n),i=1,4)
    endif
  enddo
  write(6,'(''reaxff1_under kcal '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),reaxff1_under(n)
    endif
  enddo
  write(6,'(''reaxff1_lonepair kcal '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),(reaxff1_lonepair(i,n),i=1,2)
    endif
  enddo
  write(6,'(''reaxff1_angle '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),(reaxff1_angle(i,n),i=1,2)
    endif
  enddo
  write(6,'(''reaxff1_morse kcal '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),(reaxff1_morse(i,n),i=1,4)
    endif
  enddo
  write(6,'(''#'')')
  write(6,'(''#  Element parameters '')')
  write(6,'(''#'')')
  write(6,'(''reaxff_chi  '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),reaxff_chi(n)
    endif
  enddo
  write(6,'(''reaxff_mu  '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),reaxff_mu(n)
    endif
  enddo
  write(6,'(''reaxff_gamma  '')')
  do n = 1,nspec
    if (n.ne.nx) then
      write(6,'(a2,'' core '',4(f8.4,1x))') symbol(n),reaxff_gamma(n)
    endif
  enddo
  write(6,'(''#'')')
  write(6,'(''#  Bond parameters '')')
  write(6,'(''#'')')
  first = .true.
  do n = 1,nbond
    if (reaxff2_bo(7,n).gt.0.001d0.and.reaxff2_bo(8,n).gt.0.001d0) then
      if (first) then
        write(6,'(''reaxff2_bo over bo13'')')
        first = .false.
      endif
      n1 = nbondspec(1,n)
      n2 = nbondspec(2,n)
      if (n1.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n2.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n1.gt.0.and.n2.gt.0) then
        write(6,'(a2,'' core '',a2,'' core '',6(f8.4,1x))') symbol(n1),symbol(n2),(reaxff2_bo(i,n),i=1,6)
      endif
    endif
  enddo
  first = .true.
  do n = 1,nbond
    if (reaxff2_bo(7,n).gt.0.001d0.and.reaxff2_bo(8,n).le.0.001d0) then
      if (first) then
        write(6,'(''reaxff2_bo bo13'')')
        first = .false.
      endif
      n1 = nbondspec(1,n)
      n2 = nbondspec(2,n)
      if (n1.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n2.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n1.gt.0.and.n2.gt.0) then
        write(6,'(a2,'' core '',a2,'' core '',6(f8.4,1x))') symbol(n1),symbol(n2),(reaxff2_bo(i,n),i=1,6)
      endif
    endif
  enddo
  first = .true.
  do n = 1,nbond
    if (reaxff2_bo(7,n).le.0.001d0.and.reaxff2_bo(8,n).gt.0.001d0) then
      if (first) then
        write(6,'(''reaxff2_bo over'')')
        first = .false.
      endif
      n1 = nbondspec(1,n)
      n2 = nbondspec(2,n)
      if (n1.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n2.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n1.gt.0.and.n2.gt.0) then
        write(6,'(a2,'' core '',a2,'' core '',6(f8.4,1x))') symbol(n1),symbol(n2),(reaxff2_bo(i,n),i=1,6)
      endif
    endif
  enddo
  first = .true.
  do n = 1,nbond
    if (reaxff2_bo(7,n).le.0.001d0.and.reaxff2_bo(8,n).le.0.001d0) then
      if (first) then
        write(6,'(''reaxff2_bo '')')
        first = .false.
      endif
      n1 = nbondspec(1,n)
      n2 = nbondspec(2,n)
      if (n1.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n2.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n1.gt.0.and.n2.gt.0) then
        write(6,'(a2,'' core '',a2,'' core '',6(f8.4,1x))') symbol(n1),symbol(n2),(reaxff2_bo(i,n),i=1,6)
      endif
    endif
  enddo
  write(6,'(''reaxff2_bond kcal '')')
  do n = 1,nbond
    n1 = nbondspec(1,n)
    n2 = nbondspec(2,n)
    if (n1.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
      stop
    endif
    if (n2.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
      stop
    endif
    if (n1.gt.0.and.n2.gt.0) then
      write(6,'(a2,'' core '',a2,'' core '',6(f8.4,1x))') symbol(n1),symbol(n2),(reaxff2_bond(i,n),i=1,5)
    endif
  enddo
  write(6,'(''reaxff2_over '')')
  do n = 1,nbond
    n1 = nbondspec(1,n)
    n2 = nbondspec(2,n)
    if (n1.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in over parameters! '')')
      stop
    endif
    if (n2.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in over parameters! '')')
      stop
    endif
    if (n1.gt.0.and.n2.gt.0) then
      write(6,'(a2,'' core '',a2,'' core '',6(f8.4,1x))') symbol(n1),symbol(n2),reaxff2_over(n)
    endif
  enddo
  first = .true.
  do n = 1,nbond
    if (reaxff2_bo(9,n).gt.0.0d0) then
      if (first) then
        write(6,'(''reaxff2_pen kcal'')')
        first = .false.
      endif
      n1 = nbondspec(1,n)
      n2 = nbondspec(2,n)
      if (n1.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n2.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in bond parameters! '')')
        stop
      endif
      if (n1.gt.0.and.n2.gt.0) then
        write(6,'(a2,'' core '',a2,'' core '',3(f8.4,1x))') symbol(n1),symbol(n2),reaxff2_bo(9,n),reaxff_par(14),1.0d0
      endif
    endif
  enddo
  write(6,'(''reaxff2_morse kcal '')')
  do n = 1,nod
    n1 = nodspec(1,n)
    n2 = nodspec(2,n)
    if (n1.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in 2-body morse parameters! '')')
      stop
    endif
    if (n2.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in 2-body morse parameters! '')')
      stop
    endif
    if (n1.gt.0.and.n2.gt.0) then
      write(6,'(a2,'' core '',a2,'' core '',6(f8.4,1x))') symbol(n1),symbol(n2),(reaxff2_morse(i,n),i=1,6)
    endif
  enddo
  write(6,'(''#'')')
  write(6,'(''#  Angle parameters '')')
  write(6,'(''#'')')
  write(6,'(''reaxff3_angle kcal '')')
  do n = 1,nangle
    n1 = nanglespec(1,n)
    n2 = nanglespec(2,n)
    n3 = nanglespec(3,n)
    if (n1.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in angle parameters! '')')
      stop
    endif
    if (n2.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in angle parameters! '')')
      stop
    endif
    if (n3.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in angle parameters! '')')
      stop
    endif
    if (reaxff3_angle(2,n).gt.0.0d0) then
      write(6,'(3(a2,'' core ''),6(f8.4,1x))') symbol(n1),symbol(n2),symbol(n3),(reaxff3_angle(i,n),i=1,5)
    endif
  enddo
  write(6,'(''reaxff3_penalty kcal '')')
  do n = 1,nangle
    n1 = nanglespec(1,n)
    n2 = nanglespec(2,n)
    n3 = nanglespec(3,n)
    if (n1.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in angle penalty parameters! '')')
      stop
    endif
    if (n2.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in angle penalty parameters! '')')
      stop
    endif
    if (n3.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in angle penalty parameters! '')')
      stop
    endif
    write(6,'(3(a2,'' core ''),6(f8.4,1x))') symbol(n1),symbol(n2),symbol(n3),reaxff3_penalty(n)
  enddo
  write(6,'(''reaxff3_conjugation kcal '')')
  do n = 1,nangle
    if (abs(reaxff3_conj(n)).gt.1.0d-4) then
      n1 = nanglespec(1,n)
      n2 = nanglespec(2,n)
      n3 = nanglespec(3,n)
      if (n1.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in angle conjugation parameters! '')')
        stop
      endif
      if (n2.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in angle conjugation parameters! '')')
        stop
      endif
      if (n3.gt.nspec) then
        write(6,'('' ERROR: Species number out of bounds in angle conjugation parameters! '')')
        stop
      endif
      write(6,'(3(a2,'' core ''),4(f8.4,1x))') symbol(n1),symbol(n2),symbol(n3),reaxff3_conj(n), &
        reaxff_par(3),reaxff_par(39),reaxff_par(31)
    endif
  enddo
  write(6,'(''#'')')
  write(6,'(''#  Hydrogen bond parameters '')')
  write(6,'(''#'')')
  write(6,'(''reaxff3_hbond kcal '')')
  do n = 1,nhb
    n1 = nhbspec(1,n)
    n2 = nhbspec(2,n)
    n3 = nhbspec(3,n)
    if (n1.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in hydrogen bond parameters! '')')
      stop
    endif
    if (n2.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in hydrogen bond parameters! '')')
      stop
    endif
    if (n3.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in hydrogen bond parameters! '')')
      stop
    endif
    write(6,'(3(a2,'' core ''),4(f8.4,1x))') symbol(n1),symbol(n2),symbol(n3),(reaxff3_hbond(i,n),i=1,4)
  enddo
  write(6,'(''#'')')
  write(6,'(''#  Torsion parameters '')')
  write(6,'(''#'')')
  write(6,'(''reaxff4_torsion kcal '')')
  do n = 1,ntors
    n1 = ntorsspec(1,n)
    n2 = ntorsspec(2,n)
    n3 = ntorsspec(3,n)
    n4 = ntorsspec(4,n)
    if (n1.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in torsion parameters! '')')
      stop
    endif
    if (n2.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in torsion parameters! '')')
      stop
    endif
    if (n3.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in torsion parameters! '')')
      stop
    endif
    if (n4.gt.nspec) then
      write(6,'('' ERROR: Species number out of bounds in torsion parameters! '')')
      stop
    endif
    write(6,'(4(a2,'' core ''),6(f8.4,1x))') symbol(n1),symbol(n2),symbol(n3),symbol(n4),(reaxff4_torsion(j,n),j=1,5)
  enddo

end program reaxff2lib
