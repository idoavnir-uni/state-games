FAMOUS_NAMES = [
    # Tech Leaders
    "Elon Musk", "Bill Gates", "Steve Jobs", "Mark Zuckerberg", "Larry Page",
    "Sergey Brin", "Tim Cook", "Satya Nadella", "Sundar Pichai", "Jack Dorsey",
    "Peter Thiel", "Reid Hoffman", "Marc Andreessen", "Travis Kalanick", "Brian Chesky",
    "Jensen Huang", "Lisa Su", "Pat Gelsinger", "Sheryl Sandberg", "Marissa Mayer",
    "Susan Wojcicki", "Ginni Rometty", "Meg Whitman", "Safra Catz", "Whitney Wolfe",
    
    # Actors/Actresses
    "Tom Hanks", "Leonardo DiCaprio", "Brad Pitt", "George Clooney", "Johnny Depp",
    "Robert Downey", "Denzel Washington", "Morgan Freeman", "Samuel Jackson", "Will Smith",
    "Tom Cruise", "Matt Damon", "Christian Bale", "Joaquin Phoenix", "Ryan Gosling",
    "Chris Hemsworth", "Chris Evans", "Chris Pratt", "Keanu Reeves", "Hugh Jackman",
    "Meryl Streep", "Julia Roberts", "Sandra Bullock", "Nicole Kidman", "Cate Blanchett",
    "Angelina Jolie", "Scarlett Johansson", "Jennifer Lawrence", "Emma Stone", "Natalie Portman",
    "Anne Hathaway", "Amy Adams", "Jessica Chastain", "Charlize Theron", "Kate Winslet",
    "Margot Robbie", "Gal Gadot", "Brie Larson", "Zendaya Coleman", "Florence Pugh",
    "Harrison Ford", "Anthony Hopkins", "Al Pacino", "Robert Niro", "Jack Nicholson",
    "Clint Eastwood", "Michael Douglas", "Kevin Costner", "Richard Gere", "Dustin Hoffman",
    "Emma Watson", "Daniel Radcliffe", "Rupert Grint", "Eddie Redmayne", "Benedict Cumberbatch",
    "Tom Hiddleston", "Idris Elba", "Colin Firth", "Jude Law", "Jason Statham",
    "Dwayne Johnson", "Vin Diesel", "Jason Momoa", "Dave Bautista", "John Cena",
    "Viola Davis", "Lupita Nyongo", "Halle Berry", "Angela Bassett", "Octavia Spencer",
    "Regina King", "Taraji Henson", "Kerry Washington", "Zoe Saldana", "Thandiwe Newton",
    
    # Musicians
    "Taylor Swift", "Beyonce Knowles", "Rihanna Fenty", "Lady Gaga", "Ariana Grande",
    "Katy Perry", "Adele Adkins", "Billie Eilish", "Dua Lipa", "Selena Gomez",
    "Justin Bieber", "Ed Sheeran", "Bruno Mars", "Drake Graham", "Kanye West",
    "Jay-Z Carter", "Eminem Mathers", "Post Malone", "Travis Scott", "The Weeknd",
    "Harry Styles", "Shawn Mendes", "Charlie Puth", "Sam Smith", "John Legend",
    "Paul McCartney", "Mick Jagger", "Elton John", "Freddie Mercury", "David Bowie",
    "Prince Rogers", "Michael Jackson", "Whitney Houston", "Mariah Carey", "Celine Dion",
    "Madonna Ciccone", "Britney Spears", "Christina Aguilera", "Jennifer Lopez", "Shakira Ripoll",
    "Bob Dylan", "Bruce Springsteen", "Eric Clapton", "Jimmy Page", "Keith Richards",
    "Stevie Wonder", "Lionel Richie", "Phil Collins", "Billy Joel", "Rod Stewart",
    "Kendrick Lamar", "Chance Bennett", "Tyler Creator", "Frank Ocean", "Childish Gambino",
    "Pharrell Williams", "Timbaland Mosley", "Diplo Pentz", "Calvin Harris", "David Guetta",
    "Miley Cyrus", "Demi Lovato", "Halsey Frangipane", "Camila Cabello", "Normani Hamilton",
    "Cardi B", "Megan Stallion", "Nicki Minaj", "Iggy Azalea", "Lizzo Jefferson",
    "Olivia Rodrigo", "Doja Cat", "SZA Rowe", "Summer Walker", "HER Wilson",
    
    # Athletes
    "Michael Jordan", "LeBron James", "Kobe Bryant", "Stephen Curry", "Kevin Durant",
    "Shaquille ONeal", "Magic Johnson", "Larry Bird", "Tim Duncan", "Kareem Abdul",
    "Tom Brady", "Peyton Manning", "Aaron Rodgers", "Patrick Mahomes", "Joe Montana",
    "Jerry Rice", "Walter Payton", "Barry Sanders", "Emmitt Smith", "Lawrence Taylor",
    "Lionel Messi", "Cristiano Ronaldo", "Neymar Junior", "Kylian Mbappe", "Erling Haaland",
    "Diego Maradona", "Pele Santos", "Zinedine Zidane", "Ronaldinho Gaucho", "David Beckham",
    "Roger Federer", "Rafael Nadal", "Novak Djokovic", "Serena Williams", "Venus Williams",
    "Maria Sharapova", "Naomi Osaka", "Simona Halep", "Steffi Graf", "Martina Navratilova",
    "Tiger Woods", "Phil Mickelson", "Rory McIlroy", "Jordan Spieth", "Dustin Johnson",
    "Jack Nicklaus", "Arnold Palmer", "Gary Player", "Greg Norman", "Ernie Els",
    "Usain Bolt", "Carl Lewis", "Jesse Owens", "Michael Phelps", "Katie Ledecky",
    "Simone Biles", "Nadia Comaneci", "Mary Retton", "Gabby Douglas", "Aly Raisman",
    "Muhammad Ali", "Mike Tyson", "Floyd Mayweather", "Manny Pacquiao", "Oscar Hoya",
    "Wayne Gretzky", "Sidney Crosby", "Alex Ovechkin", "Connor McDavid", "Mario Lemieux",
    "Derek Jeter", "Babe Ruth", "Mickey Mantle", "Willie Mays", "Hank Aaron",
    "Mike Trout", "Bryce Harper", "Mookie Betts", "Shohei Ohtani", "Aaron Judge",
    
    # Politicians & World Leaders
    "Barack Obama", "Michelle Obama", "Joe Biden", "Kamala Harris", "Hillary Clinton",
    "Bill Clinton", "Donald Trump", "George Bush", "George Washington", "Abraham Lincoln",
    "Franklin Roosevelt", "John Kennedy", "Ronald Reagan", "Jimmy Carter", "Richard Nixon",
    "Angela Merkel", "Emmanuel Macron", "Boris Johnson", "Rishi Sunak", "Justin Trudeau",
    "Vladimir Putin", "Xi Jinping", "Narendra Modi", "Jacinda Ardern", "Scott Morrison",
    "Benjamin Netanyahu", "Volodymyr Zelensky", "Olaf Scholz", "Pedro Sanchez", "Mario Draghi",
    "Nelson Mandela", "Desmond Tutu", "Kofi Annan", "Ban Kimoon", "Antonio Guterres",
    "Margaret Thatcher", "Winston Churchill", "Tony Blair", "David Cameron", "Theresa May",
    "Alexandria Ocasio", "Bernie Sanders", "Elizabeth Warren", "Nancy Pelosi", "Mitch McConnell",
    "Ruth Ginsburg", "Sonia Sotomayor", "Elena Kagan", "John Roberts", "Clarence Thomas",
    
    # Scientists & Inventors
    "Albert Einstein", "Stephen Hawking", "Neil Tyson", "Carl Sagan", "Richard Feynman",
    "Nikola Tesla", "Thomas Edison", "Alexander Bell", "Marie Curie", "Isaac Newton",
    "Charles Darwin", "Galileo Galilei", "Johannes Kepler", "Niels Bohr", "Max Planck",
    "Werner Heisenberg", "Erwin Schrodinger", "Paul Dirac", "Enrico Fermi", "Robert Oppenheimer",
    "James Watson", "Francis Crick", "Rosalind Franklin", "Jonas Salk", "Louis Pasteur",
    "Alexander Fleming", "Edward Jenner", "Florence Nightingale", "Elizabeth Blackwell", "Jane Goodall",
    "Rachel Carson", "Sylvia Earle", "Mae Jemison", "Sally Ride", "Katherine Johnson",
    "Alan Turing", "Ada Lovelace", "Grace Hopper", "Tim Berners", "Vint Cerf",
    "Linus Torvalds", "Dennis Ritchie", "Ken Thompson", "Bjarne Stroustrup", "James Gosling",
    "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng", "Fei-Fei Li",
    
    # Authors & Intellectuals
    "William Shakespeare", "Jane Austen", "Charles Dickens", "Mark Twain", "Ernest Hemingway",
    "Virginia Woolf", "James Joyce", "Franz Kafka", "Leo Tolstoy", "Fyodor Dostoevsky",
    "Gabriel Marquez", "Jorge Borges", "Pablo Neruda", "Isabel Allende", "Mario Vargas",
    "Toni Morrison", "Maya Angelou", "James Baldwin", "Langston Hughes", "Zora Hurston",
    "Stephen King", "John Grisham", "Dan Brown", "James Patterson", "Dean Koontz",
    "JK Rowling", "George Martin", "Brandon Sanderson", "Patrick Rothfuss", "Neil Gaiman",
    "Agatha Christie", "Arthur Doyle", "Edgar Poe", "HP Lovecraft", "Ray Bradbury",
    "Isaac Asimov", "Arthur Clarke", "Philip Dick", "Ursula LeGuin", "Octavia Butler",
    "Noam Chomsky", "Michel Foucault", "Jacques Derrida", "Judith Butler", "Slavoj Zizek",
    "Jordan Peterson", "Sam Harris", "Richard Dawkins", "Christopher Hitchens", "Daniel Dennett",
    
    # Directors & Filmmakers
    "Steven Spielberg", "Martin Scorsese", "Francis Coppola", "Quentin Tarantino", "Christopher Nolan",
    "Stanley Kubrick", "Alfred Hitchcock", "Ridley Scott", "James Cameron", "Peter Jackson",
    "David Fincher", "Denis Villeneuve", "Guillermo Toro", "Alfonso Cuaron", "Alejandro Inarritu",
    "Coen Brothers", "Wes Anderson", "Paul Anderson", "David Lynch", "Darren Aronofsky",
    "Greta Gerwig", "Kathryn Bigelow", "Sofia Coppola", "Ava DuVernay", "Patty Jenkins",
    "Chloe Zhao", "Lulu Wang", "Emerald Fennell", "Olivia Wilde", "Regina King",
    "Spike Lee", "Jordan Peele", "Ryan Coogler", "Barry Jenkins", "Steve McQueen",
    "Bong Joonho", "Park Chanwook", "Wong Karwai", "Ang Lee", "Zhang Yimou",
    "Hayao Miyazaki", "Akira Kurosawa", "Satoshi Kon", "Mamoru Hosoda", "Makoto Shinkai",
    "Federico Fellini", "Ingmar Bergman", "Andrei Tarkovsky", "Werner Herzog", "Lars Trier",
    
    # TV Personalities & Talk Show Hosts
    "Oprah Winfrey", "Ellen DeGeneres", "Jimmy Fallon", "Jimmy Kimmel", "Stephen Colbert",
    "Trevor Noah", "John Oliver", "Seth Meyers", "Conan OBrien", "James Corden",
    "David Letterman", "Jay Leno", "Johnny Carson", "Dick Cavett", "Larry King",
    "Anderson Cooper", "Rachel Maddow", "Tucker Carlson", "Sean Hannity", "Chris Wallace",
    "Barbara Walters", "Diane Sawyer", "Katie Couric", "Robin Roberts", "Gayle King",
    "Ryan Seacrest", "Simon Cowell", "Howard Stern", "Joe Rogan", "Marc Maron",
    
    # Comedians
    "Jerry Seinfeld", "Chris Rock", "Dave Chappelle", "Eddie Murphy", "Richard Pryor",
    "George Carlin", "Robin Williams", "Jim Carrey", "Adam Sandler", "Will Ferrell",
    "Steve Carell", "Tina Fey", "Amy Poehler", "Melissa McCarthy", "Kristen Wiig",
    "Maya Rudolph", "Kate McKinnon", "Cecily Strong", "Leslie Jones", "Aidy Bryant",
    "Kevin Hart", "Gabriel Iglesias", "Trevor Noah", "Hasan Minhaj", "Aziz Ansari",
    "Kumail Nanjiani", "Mindy Kaling", "Ali Wong", "Awkwafina Lum", "Ken Jeong",
    "Bill Murray", "Dan Aykroyd", "John Belushi", "Chevy Chase", "Steve Martin",
    "John Mulaney", "Bo Burnham", "Pete Davidson", "Ricky Gervais", "Russell Brand",
    
    # Business Leaders
    "Warren Buffett", "Charlie Munger", "Jamie Dimon", "Lloyd Blankfein", "Ray Dalio",
    "Carl Icahn", "George Soros", "Michael Bloomberg", "Rupert Murdoch", "Bernard Arnault",
    "Francois Pinault", "Amancio Ortega", "Mukesh Ambani", "Gautam Adani", "Jack Ma",
    "Pony Ma", "Robin Li", "Lei Jun", "Richard Liu", "Zhang Yiming",
    "Larry Ellison", "Michael Dell", "Steve Ballmer", "Paul Allen", "Gordon Moore",
    "Bob Iger", "Reed Hastings", "Ted Sarandos", "Brian Roberts", "David Zaslav",
    "Mary Barra", "James Farley", "Carlos Tavares", "Herbert Diess", "Akio Toyoda",
    "Howard Schultz", "Brian Niccol", "Chris Kempczinski", "Dara Khosrowshahi", "Tony Xu",
    "Richard Branson", "Michael OLeary", "Oscar Munoz", "Doug Parker", "Ed Bastian",
    "Andy Jassy", "Doug McMillon", "Brian Cornell", "Corie Barry", "Chip Bergh",
    
    # Fashion & Models
    "Coco Chanel", "Giorgio Armani", "Ralph Lauren", "Calvin Klein", "Tommy Hilfiger",
    "Donatella Versace", "Stella McCartney", "Vera Wang", "Michael Kors", "Marc Jacobs",
    "Karl Lagerfeld", "Tom Ford", "Alexander McQueen", "John Galliano", "Vivienne Westwood",
    "Naomi Campbell", "Cindy Crawford", "Claudia Schiffer", "Linda Evangelista", "Christy Turlington",
    "Tyra Banks", "Heidi Klum", "Gisele Bundchen", "Kate Moss", "Adriana Lima",
    "Kendall Jenner", "Gigi Hadid", "Bella Hadid", "Kaia Gerber", "Cara Delevingne",
    "David Gandy", "Sean OPry", "Lucky Blue", "Jon Kortajarena", "Tyson Beckford",
    "Anna Wintour", "Grace Coddington", "Andre Talley", "Edward Enninful", "Hamish Bowles",
    
    # Chefs & Food Personalities
    "Gordon Ramsay", "Anthony Bourdain", "Julia Child", "Wolfgang Puck", "Thomas Keller",
    "Emeril Lagasse", "Bobby Flay", "Guy Fieri", "Rachael Ray", "Ina Garten",
    "Jamie Oliver", "Nigella Lawson", "Heston Blumenthal", "Marco White", "Fergus Henderson",
    "Rene Redzepi", "Massimo Bottura", "Alain Ducasse", "Daniel Boulud", "Eric Ripert",
    "David Chang", "Roy Choi", "Jose Andres", "Dominique Crenn", "Nancy Silverton",
    
    # Royalty & Nobility
    "Queen Elizabeth", "King Charles", "Prince William", "Prince Harry", "Kate Middleton",
    "Meghan Markle", "Princess Diana", "Prince Philip", "Princess Anne", "Princess Beatrice",
    "Prince Albert", "Princess Charlene", "King Felipe", "Queen Letizia", "King Willem",
    "Queen Maxima", "King Carl", "Queen Silvia", "Crown Princess", "Prince Frederik",
    
    # Activists & Humanitarians
    "Martin Luther", "Rosa Parks", "Malcolm X", "John Lewis", "Jesse Jackson",
    "Al Sharpton", "Cornel West", "Angela Davis", "Stokely Carmichael", "Huey Newton",
    "Gloria Steinem", "Betty Friedan", "Ruth Ginsburg", "Malala Yousafzai", "Greta Thunberg",
    "Mahatma Gandhi", "Dalai Lama", "Mother Teresa", "Desmond Tutu", "Wangari Maathai",
    "Cesar Chavez", "Dolores Huerta", "Harvey Milk", "Marsha Johnson", "Sylvia Rivera",
    "Jane Fonda", "Robert Redford", "Leonardo DiCaprio", "Mark Ruffalo", "Emma Watson",
    
    # Astronauts & Space
    "Neil Armstrong", "Buzz Aldrin", "Michael Collins", "John Glenn", "Alan Shepard",
    "Sally Ride", "Mae Jemison", "Christina Koch", "Peggy Whitson", "Scott Kelly",
    "Mark Kelly", "Chris Hadfield", "Tim Peake", "Samantha Cristoforetti", "Thomas Pesquet",
    "Yuri Gagarin", "Valentina Tereshkova", "Alexei Leonov", "Sergei Krikalev", "Gennady Padalka",
    "Yang Liwei", "Liu Yang", "Wang Yaping", "Zhai Zhigang", "Nie Haisheng",
    
    # YouTubers & Internet Personalities
    "PewDiePie Kjellberg", "MrBeast Donaldson", "Logan Paul", "Jake Paul", "David Dobrik",
    "Emma Chamberlain", "James Charles", "Jeffree Star", "Nikkie Tutorials", "Jackie Aina",
    "Casey Neistat", "Philip DeFranco", "Rhett McLaughlin", "Link Neal", "Markiplier Fischbach",
    "Jacksepticeye McLoughlin", "Ninja Blevins", "Shroud Grzesiek", "Pokimane Anys", "Valkyrae Hofstetter",
    "Hank Green", "John Green", "Simone Giertz", "Marques Brownlee", "Linus Sebastian",
    
    # Architects & Designers
    "Frank Lloyd", "Frank Gehry", "Zaha Hadid", "Norman Foster", "Renzo Piano",
    "Tadao Ando", "Rem Koolhaas", "Bjarke Ingels", "Santiago Calatrava", "Daniel Libeskind",
    "Jony Ive", "Dieter Rams", "Charles Eames", "Ray Eames", "Philippe Starck",
    
    # Additional Artists & Painters
    "Pablo Picasso", "Salvador Dali", "Vincent Gogh", "Claude Monet", "Leonardo Vinci",
    "Michelangelo Buonarroti", "Rembrandt Rijn", "Andy Warhol", "Jean Basquiat", "Keith Haring",
    "Frida Kahlo", "Georgia OKeeffe", "Yayoi Kusama", "Marina Abramovic", "Cindy Sherman",
    "Banksy Artist", "Damien Hirst", "Jeff Koons", "Ai Weiwei", "Takashi Murakami",
    
    # Philosophers (Historical & Modern)
    "Plato Athens", "Aristotle Stagira", "Socrates Athens", "Confucius Kong", "Laozi China",
    "Immanuel Kant", "Friedrich Nietzsche", "Jean Sartre", "Simone Beauvoir", "Hannah Arendt",
    "Bertrand Russell", "Ludwig Wittgenstein", "Martin Heidegger", "Edmund Husserl", "Maurice Merleau",
    
    # Classical Musicians & Composers
    "Ludwig Beethoven", "Wolfgang Mozart", "Johann Bach", "Frederic Chopin", "Franz Liszt",
    "Johannes Brahms", "Richard Wagner", "Pyotr Tchaikovsky", "Igor Stravinsky", "Claude Debussy",
    "Yo-Yo Ma", "Itzhak Perlman", "Lang Lang", "Joshua Bell", "Hilary Hahn",
    "Placido Domingo", "Luciano Pavarotti", "Andrea Bocelli", "Maria Callas", "Renee Fleming",
    
    # More Contemporary Musicians
    "John Mayer", "Jason Mraz", "Jack Johnson", "John Legend", "Alicia Keys",
    "Lauryn Hill", "Erykah Badu", "India Arie", "Jill Scott", "Maxwell Singer",
    "Usher Raymond", "Chris Brown", "Jason Derulo", "Ne-Yo Smith", "Trey Songz",
    "Justin Timberlake", "Nick Jonas", "Joe Jonas", "Kevin Jonas", "Zayn Malik",
    "Liam Payne", "Niall Horan", "Louis Tomlinson", "Gwen Stefani", "Fergie Duhamel",
    
    # Reality TV Stars
    "Kim Kardashian", "Kourtney Kardashian", "Khloe Kardashian", "Kylie Jenner", "Kris Jenner",
    "Paris Hilton", "Nicole Richie", "Lauren Conrad", "Kristin Cavallari", "Heidi Montag",
    "Snooki Polizzi", "Jwoww Farley", "Mike Sorrentino", "Pauly DelVecchio", "Vinny Guadagnino",
    
    # Legendary Athletes (Additional)
    "Jackie Robinson", "Joe DiMaggio", "Ted Williams", "Lou Gehrig", "Roberto Clemente",
    "Sandy Koufax", "Nolan Ryan", "Roger Clemens", "Pedro Martinez", "Randy Johnson",
    "Wilt Chamberlain", "Bill Russell", "Oscar Robertson", "Jerry West", "Julius Erving",
    "Charles Barkley", "Karl Malone", "John Stockton", "Patrick Ewing", "Scottie Pippen",
    "Jim Brown", "Dick Butkus", "Ray Lewis", "Ed Reed", "Troy Polamalu",
    "Brett Favre", "Drew Brees", "Eli Manning", "Russell Wilson", "Lamar Jackson",
    "Thierry Henry", "Dennis Bergkamp", "Patrick Vieira", "Frank Lampard", "Steven Gerrard",
    "Wayne Rooney", "Sergio Aguero", "Kevin Bruyne", "Virgil Dijk", "Mohamed Salah",
    
    # Gaming & Esports
    "Faker Lee", "Dendi Ishutin", "S1mple Kostyliev", "ZywOo Herbaut", "Dev1ce Reedtz",
    "Bugha Giersdorf", "Tfue Tenney", "Mongraal Bennett", "Clix Deyerin", "Unknown Army",
    "Daigo Umehara", "Tokido Taniguchi", "SonicFox McLean", "Hungrybox Debiedma", "Mango Marquez",
    
    # Podcasters & Media
    "Joe Rogan", "Tim Ferriss", "Lex Fridman", "Jordan Harbinger", "Lewis Howes",
    "Gary Vaynerchuk", "Tony Robbins", "Brene Brown", "Malcolm Gladwell"
]