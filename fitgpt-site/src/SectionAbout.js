import Profile from './Profile';

function SectionAbout() {
	const profiles = [
		[
			"berrybear06",
			<>
			<p>Aspiring programmer and engineer playing around with AI.</p>
			<a className="link" href="https://github.com/berrybear06">Find me on GitHub</a>
			</>,
			"berrybearpfp.webp"
		],
		[
			"qwertyblurby",
			<>
			<p>I'm Erik Shen</p>
			<a className='link' href="https://github.com/qwertyblurby">Find me on GitHUb</a>
			</>,
			"https://img.allfootballapp.com/www/M00/19/31/720x-/-/-/CgAGVmDIU3iAKA7SAAFnj6PAUHg046.jpg.webp"
		],
		[
			"chickeno7",
			<>
			<p>I know how to code on occasion.</p>
			<a className="link" href="https://github.com/chicken07">Find me on GitHub</a>
			</>,
			"https://cdn.discordapp.com/avatars/434530835962658817/c9b905a1828187809f6c94a91d9792ce.webp"
		],
		[
			"cronchbird",
			"cronch bio",
			"https://avatars.githubusercontent.com/u/120995210?s=256&v=4"
		],
	]
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">About Us</h2>
			<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
				{profiles.map(([name, bio, imageSrc]) => (
					<Profile
						key={name}
						name={name}
						bio={bio}
						imageSrc={imageSrc}
					/>
				))}
			</div>
		</div>
	);
}

export default SectionAbout;
