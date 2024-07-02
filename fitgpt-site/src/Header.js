import MenuIcon from './assets/icons/MenuIcon';
// import MountainIcon from './assets/icons/MountainIcon';

function Header() {
	return (
		<header id="hero" className="sticky top-0 z-50 p-4 md:px-12 lg:px-16 bg-gray-950 text-gray-50">
			<div className="navbar bg-base-100 rounded-lg">
				<div className="navbar-start">
					<div className="dropdown">
						<div tabIndex={0} role="button" className="btn btn-ghost lg:hidden">
							<MenuIcon className="h-5 w-5"/>
						</div>
						
						<ul
							tabIndex={0}
							className="menu menu-sm dropdown-content bg-base-100 rounded-box z-[1] mt-3 w-52 p-2 shadow"
						>
							<li><a href="#caps">Features</a></li>
							<li><a href="#recs">Demo</a></li>
							<li><a href="#tech">Tech</a></li>
							<li><a href="#about">About Us</a></li>
						</ul>
					</div>
					
					{/* <MountainIcon className="h-8 w-8 mx-4 hidden lg:block" /> */}
					<a href="#home" className="btn btn-ghost text-xl">FitGPT</a>
				</div>
				
				<div className="navbar-center hidden lg:flex">
					<ul className="menu menu-horizontal px-1 space-x-1">
						<li><a href="#demo">Demo</a></li>
						<li><a href="#caps">Features</a></li>
						<li><a href="#tech">Tech</a></li>
						<li><a href="#about">About Us</a></li>
					</ul>
				</div>
				
				<div className="navbar-end">
					<a href="#demo"
						className="btn dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700"
					>
						Try it Now
					</a>
				</div>
			</div>
		</header>
	)
}

export default Header;
